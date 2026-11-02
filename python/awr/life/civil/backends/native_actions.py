"""Causal authored-action execution within an existing py_trees service step.

This is an activity/animation adapter, not a pose or motion generator. Navigation
must already have stopped. Only verified anchored-action clips are supported;
prop, seat, partner and root-motion actions require a different executor.
"""

import hashlib
import json
import math
from pathlib import Path

from awr.life.common.schema import schema_id


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class NativeActions:
    def __init__(self, config, actors):
        self.config = config
        self.assignments = dict(config["actor_packages"])
        if set(self.assignments) != set(actors):
            raise ValueError("Action package required for every actor")
        self.packages = {}
        if set(config["packages"]) != set(self.assignments.values()):
            raise ValueError("Action registry mismatch")
        for sha, row in config["packages"].items():
            folder = Path(row["directory"])
            path = folder / "manifest.json"
            if digest(path) != sha:
                raise ValueError("Action package manifest mismatch")
            spec = json.loads(path.read_text())
            if (
                schema_id(spec.get("schema")) != "awr.life.ozz_authored_package.v1"
                or digest(folder / "authored-trs.bin") != spec["package_sha256"]
            ):
                raise ValueError("Action package payload mismatch")
            self.packages[sha] = spec
        self.rules = dict(config["purposes"])
        self.sessions = {}
        self.current = {}
        self.settle_s = float(config.get("settle_s", 0.3))
        self.blend_s = float(config.get("blend_s", 0.35))
        if not all(math.isfinite(v) and v > 0 for v in (self.settle_s, self.blend_s)):
            raise ValueError("Positive finite action timing required")
        for purpose, rule in self.rules.items():
            if (
                not purpose
                or rule.get("executor") != "standing_observation"
                or not rule.get("requires")
                or not rule.get("clips")
                or len(set(rule["clips"])) != len(rule["clips"])
            ):
                raise ValueError("Explicit standing-observation semantics required")
            if not all(
                math.isfinite(rule.get(k, math.nan)) and rule[k] > 0
                for k in ("max_local_translation_m", "max_local_yaw_rad")
            ):
                raise ValueError("Explicit finite local action envelope required")
            for spec in self.packages.values():
                for name in rule["clips"]:
                    clip = spec["clips"].get(name, {})
                    if (
                        not math.isfinite(clip.get("duration_s", float("nan")))
                        or clip["duration_s"] <= 2 * self.blend_s
                        or clip.get("root_mode") != "local_pose"
                        or not 0 <= clip.get("root_translation_max_m", math.inf) <= rule["max_local_translation_m"]
                        or not 0 <= clip.get("root_yaw_max_rad", math.inf) <= rule["max_local_yaw_rad"]
                    ):
                        raise ValueError("Action needs a verified stationary-root authored clip: " + name)

    def tick(self, context, state, place):
        """Return true when this action is done; caller still owns service completion."""
        c = context
        now = c.runtime.crowd.time
        actor = c.actor
        rule = self.rules.get(c.intent.purpose)
        if rule is None:
            self.current.pop(actor, None)
            return True
        if not set(rule["requires"]) <= set(place.affordances):
            raise ValueError("Action place lacks required affordances")
        if not place.opens_s <= now < place.closes_s:
            raise RuntimeError("Action place closed during execution")
        key = (actor, c.intent.id, c.intent.index)
        session = self.sessions.get(key)
        if session is None:
            # Stable per-intent choice; no horizon, future pose or per-person path.
            choice = int(hashlib.sha256((str(c.runtime.director.seed) + "|" + str(key)).encode()).hexdigest(), 16)
            clip = rule["clips"][choice % len(rule["clips"])]
            package = self.assignments[actor]
            session = dict(
                clip=clip,
                package_sha256=package,
                duration_s=self.packages[package]["clips"][clip]["duration_s"],
                stable_since=None,
                start_s=None,
                completed=False,
            )
            self.sessions[key] = session
        self.current[actor] = key
        if session["completed"]:
            return True
        valid = state.goal is None and state.speed <= 0.03
        if session["start_s"] is None:
            if not valid:
                session["stable_since"] = None
                return False
            if session["stable_since"] is None:
                session["stable_since"] = now
            if now - session["stable_since"] + 1e-9 < self.settle_s:
                return False
            session["start_s"] = now
            c.emit(
                "native_action_started",
                clip=session["clip"],
                package_sha256=session["package_sha256"],
                duration_s=session["duration_s"],
                blend_s=self.blend_s,
                place=place.id,
                executor=rule["executor"],
                requires=rule["requires"],
                max_local_translation_m=rule["max_local_translation_m"],
                max_local_yaw_rad=rule["max_local_yaw_rad"],
            )
        elif not valid:
            c.emit("native_action_interrupted", clip=session["clip"], reason="movement_during_stationary_action")
            raise RuntimeError("Movement during stationary authored action")
        if now - session["start_s"] + 1e-9 >= session["duration_s"]:
            session["completed"] = True
            c.emit(
                "native_action_completed",
                clip=session["clip"],
                package_sha256=session["package_sha256"],
                start_s=session["start_s"],
                duration_s=session["duration_s"],
            )
            return True
        return False

    def sample(self, actor, intent_id):
        key = self.current.get(actor)
        if key is None or key[1] != intent_id:
            return None
        s = self.sessions[key]
        if s["start_s"] is None:
            return None
        return dict(
            clip=s["clip"],
            package_sha256=s["package_sha256"],
            start_s=s["start_s"],
            duration_s=s["duration_s"],
            blend_s=self.blend_s,
            completed=s["completed"],
        )
