"""py_trees execution of existing Director intents on native DetourCrowd.

Director owns purposes, reservations and completion evidence. Native Detour owns
movement; py_trees owns ordered execution. This module neither selects random
destinations nor advances world time. Unsupported steps fail before movement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import py_trees

from awr.life.civil.activity.director import Route

Status = py_trees.common.Status


class NativeRoutes:
    """Static native route evidence for Director; no Python grid search."""

    def __init__(self, crowd, height, *, speed=1.3):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Positive finite walking speed required")
        self.crowd, self.height, self.speed = crowd, height, speed
        self.queries = 0

    def __call__(self, mode, start, end, entity):
        if mode != "walk":
            return None
        self.queries += 1
        # Use actual native height at the actor position; destination height is
        # supplied by the explicit environment adapter, not a guessed flat floor.
        state = self.crowd.state(entity)
        if math.dist(start, state.position[:2]) < 1e-5:
            origin = state.position
        else:
            origin = (*start, float(self.height(start)))
        destination = (*end, float(self.height(end)))
        points = self.crowd.path(origin, destination)
        if points is None:
            return None
        length = sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:], strict=False))
        # Endpoint projection is bounded by the separately declared native snap
        # contract. Count both connectors rather than silently shortening cost.
        length += math.dist(start, points[0][:2]) + math.dist(end, points[-1][:2])
        evidence = hashlib.sha256(
            json.dumps(
                dict(mesh=self.crowd.provenance, points=points, requested=[origin, destination]), sort_keys=True
            ).encode()
        ).hexdigest()
        return Route(length, length / self.speed, "native-detour:" + evidence)


@dataclass
class ExecutionContext:
    runtime: object
    actor: str
    intent: object
    failed: bool = False
    reason: str | None = None
    phase: str = "selected"

    def emit(self, kind, **data):
        self.runtime.director.emit(
            kind, self.actor, self.runtime.crowd.time, intent_id=self.intent.id, purpose=self.intent.purpose, **data
        )

    def fail(self, reason):
        if not self.failed:
            self.failed = True
            self.reason = reason
            self.phase = "failed"
            self.runtime.crowd.stop(self.actor)
            self.runtime.director.pause(self.actor, self.runtime.crowd.time)
            self.emit("native_execution_failed", reason=reason, step_index=self.intent.index)
        return Status.FAILURE


class _ExecuteStep(py_trees.behaviour.Behaviour):
    def __init__(self, context, index):
        super().__init__(name=f"{context.actor}/{context.intent.id}/{index}")
        self.ctx, self.index = context, index
        self.stopping = False

    def initialise(self):
        c = self.ctx
        r = c.runtime
        self.started = r.crowd.time
        self.progress_at = self.started
        self.last_position = r.crowd.state(c.actor).position
        self.stopping = False
        step = c.intent.steps[self.index]
        c.phase = step.kind
        c.emit("native_step_started", step_index=self.index, step=step.kind)
        if step.kind == "walk":
            goal = (*step.xy, float(r.height(step.xy)))
            try:
                if not r.crowd.request(c.actor, goal):
                    c.fail("native_destination_unreachable")
                else:
                    self.accepted_goal = r.crowd.state(c.actor).goal
                    c.emit("native_goal_accepted", requested=list(goal), accepted=list(self.accepted_goal))
            except RuntimeError as error:
                c.fail("native_request_error: " + str(error))

    def update(self):
        c = self.ctx
        r = c.runtime
        if c.failed:
            return Status.FAILURE
        if r.director.active.get(c.actor) is not c.intent or c.intent.index != self.index:
            return c.fail("intent_ownership_changed")
        if c.intent.paused_at is not None:
            return Status.RUNNING
        state = r.crowd.state(c.actor)
        step = c.intent.steps[self.index]
        now = r.crowd.time
        if state.motion_state != 1:
            return c.fail("native_actor_not_walking_state")
        if step.kind == "walk":
            if state.partial or state.target_state == 1:
                return c.fail("native_path_failed_or_partial")
            distance = math.dist(state.position[:2], step.xy)
            if not self.stopping:
                if state.goal != self.accepted_goal:
                    return c.fail("native_goal_changed")
                if distance <= r.arrival_radius:
                    r.crowd.stop(c.actor)
                    self.stopping = True
                    c.emit("native_arrival_braking", distance_m=distance)
            if math.dist(state.position[:2], self.last_position[:2]) > 0.2:
                self.last_position = state.position
                self.progress_at = now
            if now - self.started > r.max_walk_s:
                return c.fail("walk_deadline_exceeded")
            if now - self.progress_at > r.max_stall_s:
                return c.fail("walk_stalled")
            if not self.stopping:
                return Status.RUNNING
            if state.speed <= 0.08 and distance > r.arrival_radius:
                return c.fail("stopped_outside_arrival_region")
        else:
            if state.goal is not None:
                return c.fail("movement_goal_present_during_service")
            if math.dist(state.position[:2], step.xy) > r.arrival_radius:
                return c.fail("service_left_arrival_region")
        try:
            if step.kind == "activity" and r.actions is not None:
                if not r.actions.tick(c, state, r.director.places[c.intent.place]):
                    return Status.RUNNING
            acknowledged = r.director.acknowledge(
                c.actor, now, state.position[:2], state.speed, radius=r.arrival_radius
            )
        except RuntimeError as error:
            return c.fail("activity_acknowledgement_failed: " + str(error))
        if acknowledged:
            c.emit(
                "native_step_acknowledged",
                step_index=self.index,
                step=step.kind,
                position=list(state.position),
                speed_mps=state.speed,
            )
            return Status.SUCCESS
        return Status.RUNNING


class NativeActivities:
    """Execute supported intents; expose failures without changing their purpose.

    A failed actor stays paused with its evidence/reservations retained. It is
    never marked serviced or silently assigned another intent. World stepping
    and other actors can continue; external policy must explicitly resolve it.
    """

    def __init__(
        self, director, crowd, height, *, arrival_radius=0.5, max_walk_s=180.0, max_stall_s=30.0, actions=None
    ):
        if any(not math.isfinite(x) or x <= 0 for x in (arrival_radius, max_walk_s, max_stall_s)):
            raise ValueError("Finite positive execution thresholds required")
        if not math.isclose(director.place_arrival_radius_m, arrival_radius):
            raise ValueError("Director and native executor must share place arrival semantics")
        self.director, self.crowd, self.height = director, crowd, height
        self.arrival_radius, self.max_walk_s, self.max_stall_s = arrival_radius, max_walk_s, max_stall_s
        self.executions = {}
        self.contexts = {}
        self.actions = actions

    def tick(self, actor):
        state = self.crowd.state(actor)
        self.director.observe_departure(actor, state.position[:2])
        c = self.contexts.get(actor)
        if c and c.failed:
            return Status.FAILURE
        if actor not in self.executions:
            intent = self.director.choose(actor, self.crowd.time, state.position[:2])
            if intent is None:
                if c:
                    c.phase = "decision_wait"
                return Status.RUNNING
            c = ExecutionContext(self, actor, intent)
            self.contexts[actor] = c
            if any(s.kind not in ("walk", "activity") or s.entity != actor for s in intent.steps):
                return c.fail("unsupported_native_activity_steps")
            if state.goal is not None:
                return c.fail("unowned_movement_goal_before_intent")
            nodes = [_ExecuteStep(c, i) for i in range(len(intent.steps))]
            self.executions[actor] = py_trees.composites.Sequence(name=intent.id, memory=True, children=nodes)
        tree = self.executions[actor]
        tree.tick_once()
        if tree.status == Status.SUCCESS:
            self.contexts[actor].phase = "completed"
            del self.executions[actor]
        return tree.status

    def pause(self, actor, reason):
        if not reason:
            raise ValueError("Explicit interruption reason required")
        c = self.contexts.get(actor)
        if c is None:
            raise ValueError("No active execution")
        return c.fail("explicit_interruption: " + reason)
