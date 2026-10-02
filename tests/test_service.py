"""Service-level adjudication tests against a temporary SQLite database."""

import os
import tempfile
import unittest

from app import canonical, ed25519, merkle
from app.server import ApiError, Service, Submission
from app.storage import Store


def make_signer():
    seed, pub = ed25519.generate_keypair()

    def sign(log_id, size, ts, root):
        return ed25519.sign(
            canonical.encode_message(log_id, pub, size, ts, root), seed)

    return seed, pub, sign


class Harness:
    def __init__(self, store):
        self.svc = Service(store)
        self.seed, self.pub, self.sign = make_signer()
        self.leaves = [merkle.hash_leaf(os.urandom(8)) for _ in range(16)]
        self.roots = {n: merkle.tree_hash(self.leaves[:n])
                      for n in range(1, 17)}

    def sub(self, log_id, size, ts, root=None, proof=None,
            pub=None, signer=None):
        pub = pub or self.pub
        root = root or self.roots[size]
        proof = proof if proof is not None else []
        signer = signer or self.sign
        return Submission(pub, size, ts, root, signer(log_id, size, ts, root),
                          proof)


class ServiceRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.h = Harness(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_freeze_advance_replay_fork(self):
        h, svc = self.h, self.h.svc
        log = "L"
        status, out = svc.submit(log, h.sub(log, 3, 100))
        self.assertEqual((status, out["result"]), (201, "frozen"))

        # frozen key visible via describe
        desc = svc.describe_log(log)
        self.assertEqual(desc["tree_size"], 3)
        self.assertEqual(desc["public_key"], h.pub.hex())
        self.assertIsNone(desc["fork"])

        # valid extension
        proof = merkle.consistency_proof(3, h.leaves[:5])
        status, out = svc.submit(log, h.sub(log, 5, 200, proof=proof))
        self.assertEqual(out["result"], "trusted")
        self.assertTrue(out["applied"])

        # identical replay
        status, out = svc.submit(log, h.sub(log, 5, 200, proof=proof))
        self.assertEqual(out["result"], "already_trusted")

        # same-size different root signed by the frozen key -> fork
        rival_root = merkle.hash_leaf(b"rival")
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 5, 201, root=rival_root))
        self.assertEqual(ctx.exception.code, "fork_evidence_sealed")

        desc = svc.describe_log(log)
        self.assertEqual(desc["status"], "fork_sealed")
        self.assertEqual(desc["tree_size"], 5)
        self.assertEqual(desc["root_hash"], h.roots[5].hex())
        self.assertEqual(desc["fork"]["rival"]["root_hash"],
                         rival_root.hex())

        # second fork must not overwrite the first evidence
        rival2 = merkle.hash_leaf(b"rival2")
        with self.assertRaises(ApiError):
            svc.submit(log, h.sub(log, 5, 202, root=rival2))
        self.assertEqual(svc.describe_log(log)["fork"]["rival"]["root_hash"],
                         rival_root.hex())

        # extension blocked after seal
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(
                log, 6, 300,
                proof=merkle.consistency_proof(5, h.leaves[:6])))
        self.assertEqual(ctx.exception.code, "log_sealed")

    def test_historical_size_competing_claim(self):
        """A verified claim about an already-published size with the same
        root hash but a different signed timestamp is equivocation about
        history: seal first evidence against the original checkpoint while
        the larger trusted head stays intact."""
        h, svc = self.h, self.h.svc
        log = "H"

        # Published history: size 3 ... advanced (with a valid proof) to 5.
        status, out = svc.submit(log, h.sub(log, 3, 100))
        self.assertEqual((status, out["result"]), (201, "frozen"))
        root3 = h.roots[3]
        proof35 = merkle.consistency_proof(3, h.leaves[:5])
        status, out = svc.submit(log, h.sub(log, 5, 200, proof=proof35))
        self.assertEqual(out["result"], "trusted")

        # A stale size that was never published keeps being rejected as
        # stale, never sealed.
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 2, 50))
        self.assertEqual(ctx.exception.code, "stale_tree_size")

        # Re-signed size-3 checkpoint: same root, different millisecond
        # timestamp -> different signature over the canonical message.
        rival = h.sub(log, 3, 101, root=root3)
        self.assertNotEqual(
            rival.signature,
            h.sub(log, 3, 100, root=root3).signature)
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, rival)
        self.assertEqual(ctx.exception.code, "fork_evidence_sealed")
        self.assertEqual(
            ctx.exception.details["historical_tree_size"], 3)

        desc = svc.describe_log(log)
        # Current trusted head is untouched...
        self.assertEqual(desc["tree_size"], 5)
        self.assertEqual(desc["root_hash"], h.roots[5].hex())
        self.assertEqual(desc["public_key"], h.pub.hex())
        self.assertEqual(desc["status"], "fork_sealed")
        # ...and the evidence points at the ORIGINAL size-3 checkpoint.
        fork = desc["fork"]
        self.assertEqual(fork["trusted"]["tree_size"], 3)
        self.assertEqual(fork["trusted"]["root_hash"], root3.hex())
        self.assertEqual(fork["trusted"]["timestamp_ms"], 100)
        self.assertEqual(fork["trusted"]["signature"],
                         h.sub(log, 3, 100, root=root3).signature.hex())
        self.assertEqual(fork["rival"]["timestamp_ms"], 101)
        self.assertEqual(fork["rival"]["root_hash"], root3.hex())
        self.assertEqual(fork["rival"]["signature"], rival.signature.hex())

        # An exact replay of the original historical checkpoint still
        # replays stably...
        status, out = svc.submit(log, h.sub(log, 3, 100, root=root3))
        self.assertEqual((status, out["result"]),
                         (200, "already_trusted"))

        # ...a second competing claim does not overwrite first evidence...
        with self.assertRaises(ApiError):
            svc.submit(log, h.sub(log, 3, 102, root=root3))
        self.assertEqual(
            svc.describe_log(log)["fork"]["rival"]["timestamp_ms"], 101)

        # ...unknown old sizes stay stale...
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 2, 50))
        self.assertEqual(ctx.exception.code, "stale_tree_size")

        # ...and the sealed log can never advance.
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(
                log, 6, 300,
                proof=merkle.consistency_proof(5, h.leaves[:6])))
        self.assertEqual(ctx.exception.code, "log_sealed")

        desc = svc.describe_log(log)
        self.assertEqual(desc["tree_size"], 5)

    def test_stale_and_key_freeze(self):
        h, svc = self.h, self.h.svc
        log = "K"
        svc.submit(log, h.sub(log, 4, 100))
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 3, 90))
        self.assertEqual(ctx.exception.code, "stale_tree_size")

        _, pub2, signer2 = make_signer()
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 6, 200, pub=pub2, signer=signer2,
                                  proof=merkle.consistency_proof(
                                      4, h.leaves[:6])))
        self.assertEqual(ctx.exception.code, "public_key_frozen")

    def test_invalid_signature_rejected_at_parse(self):
        import json

        from app.server import parse_submission
        h, svc = self.h, self.h.svc
        log = "V"
        svc.submit(log, h.sub(log, 2, 100))
        payload = {
            "public_key": h.pub.hex(),
            "tree_size": 3,
            "timestamp_ms": 200,
            "root_hash": h.roots[3].hex(),
            "signature": ("00" * 64),
            "consistency": [n.hex() for n in
                            merkle.consistency_proof(2, h.leaves[:3])],
        }
        with self.assertRaises(ApiError) as ctx:
            parse_submission(log, json.dumps(payload).encode())
        self.assertEqual(ctx.exception.code, "invalid_signature")
        self.assertEqual(svc.describe_log(log)["tree_size"], 2)

    def test_bad_proof_rejected_before_write(self):
        h, svc = self.h, self.h.svc
        log = "P"
        svc.submit(log, h.sub(log, 2, 100))
        proof = merkle.consistency_proof(2, h.leaves[:6])[:-1]  # truncated
        with self.assertRaises(ApiError) as ctx:
            svc.submit(log, h.sub(log, 6, 200, proof=proof))
        self.assertEqual(ctx.exception.code, "invalid_consistency_proof")
        self.assertEqual(svc.describe_log(log)["tree_size"], 2)

    def test_unknown_log(self):
        with self.assertRaises(ApiError) as ctx:
            self.h.svc.describe_log("ghost")
        self.assertEqual(ctx.exception.http_status, 404)


if __name__ == "__main__":
    unittest.main()
