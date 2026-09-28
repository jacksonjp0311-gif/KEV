"""Bounded, receipt-bearing actions over KEV's explicit cognitive state.

This module deliberately has no dependency on a language model or on KEV's
storage implementation.  A planner can only inspect the explicit state
compartments named in :data:`EXPLICIT_STATE_KEYS`; tools must be registered;
and an executor emits an observation only after its receipt has been accepted
by the injected ledger sink.

The schema validator implements a small, documented JSON-Schema-like subset:
``type``, ``properties``, ``required``, ``additionalProperties``, ``items``,
``enum``, ``const``, numeric bounds, string/array length bounds, and ``pattern``.
It is intentionally small so the action boundary remains inspectable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


EXPLICIT_STATE_KEYS = (
    "facts",
    "goals",
    "constraints",
    "observations",
    "predictions",
    "reviewed_lessons",
)


class SchemaValidationError(ValueError):
    """A value did not satisfy a tool's declared schema."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class ReceiptWriteError(RuntimeError):
    """The ledger sink did not accept a completed action receipt.

    ``receipt`` is retained on the exception so the raw result is not lost and
    can be recovered by a caller.  No VERIFIED observation is produced when
    this exception is raised.
    """

    def __init__(self, receipt: Mapping[str, Any], cause: BaseException) -> None:
        super().__init__(f"action receipt could not be persisted: {cause}")
        self.receipt = copy.deepcopy(dict(receipt))
        self.__cause__ = cause


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for evidence hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    """Return a labelled SHA-256 digest of canonical JSON data."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _json_clone(value: Any, *, path: str = "$") -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(path, "value must be finite JSON data") from exc


def _receipt_safe(value: Any) -> Any:
    """Represent even invalid/raw values without losing failure evidence."""

    try:
        return _json_clone(value)
    except SchemaValidationError:
        return {
            "unserializable_type": type(value).__name__,
            "representation": repr(value),
        }


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type {expected!r}")


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate *value* against KEV's deliberately small schema subset.

    The function raises :class:`SchemaValidationError` at the first mismatch.
    Empty schemas accept any finite JSON value.  Schema definitions themselves
    are checked when a :class:`ToolSpec` is created.
    """

    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(path, f"must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(path, f"must be one of {schema['enum']!r}")

    declared = schema.get("type")
    if declared is not None:
        expected_types = [declared] if isinstance(declared, str) else list(declared)
        if not any(_matches_type(value, item) for item in expected_types):
            joined = " or ".join(expected_types)
            raise SchemaValidationError(path, f"expected {joined}")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaValidationError(path, f"missing required property {key!r}")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(path, "object keys must be strings")
            child_path = f"{path}.{key}"
            if key in properties:
                validate_schema(item, properties[key], child_path)
            elif additional is False:
                raise SchemaValidationError(
                    child_path, "additional property is forbidden"
                )
            elif isinstance(additional, Mapping):
                validate_schema(item, additional, child_path)

    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaValidationError(
                path, f"requires at least {schema['minItems']} items"
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(
                path, f"allows at most {schema['maxItems']} items"
            )
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaValidationError(
                path, f"requires length >= {schema['minLength']}"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(
                path, f"requires length <= {schema['maxLength']}"
            )
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaValidationError(path, f"does not match {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise SchemaValidationError(path, "number must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(path, f"must be <= {schema['maximum']}")

    # Even an empty schema accepts JSON data only.  This keeps receipts
    # canonical and prevents handlers from smuggling opaque Python objects.
    _json_clone(value, path=path)


def _check_schema_definition(schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(schema, Mapping):
        raise TypeError(f"{path} must be a mapping")
    declared = schema.get("type")
    if declared is not None:
        values = [declared] if isinstance(declared, str) else declared
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{path}.type must be a string or non-empty list")
        for item in values:
            _matches_type(None if item == "null" else _sentinel_for_type(item), item)
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise TypeError(f"{path}.properties must be a mapping")
    for name, child in properties.items():
        if not isinstance(name, str):
            raise TypeError(f"{path}.properties keys must be strings")
        _check_schema_definition(child, f"{path}.properties.{name}")
    if isinstance(schema.get("items"), Mapping):
        _check_schema_definition(schema["items"], f"{path}.items")
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, Mapping):
        _check_schema_definition(additional, f"{path}.additionalProperties")
    _json_clone(schema, path=path)


def _sentinel_for_type(kind: str) -> Any:
    sentinels = {
        "object": {},
        "array": [],
        "string": "",
        "boolean": False,
        "integer": 0,
        "number": 0.0,
        "null": None,
    }
    if kind not in sentinels:
        raise ValueError(f"unsupported schema type {kind!r}")
    return sentinels[kind]


ToolHandler = Callable[[Any], Any]
LedgerSink = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A declared tool and the schemas at its trust boundary."""

    name: str
    handler: ToolHandler
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    production_mutation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]*", self.name
        ):
            raise ValueError("tool name must match [a-z][a-z0-9_.-]*")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        _check_schema_definition(self.input_schema, "input_schema")
        _check_schema_definition(self.output_schema, "output_schema")
        object.__setattr__(
            self, "input_schema", MappingProxyType(_json_clone(self.input_schema))
        )
        object.__setattr__(
            self, "output_schema", MappingProxyType(_json_clone(self.output_schema))
        )


class ToolRegistry:
    """Explicit allowlist of tools.  A new registry contains no tools."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("only ToolSpec instances may be registered")
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    allow = register

    def unregister(self, name: str) -> ToolSpec:
        return self._tools.pop(name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"tool {name!r} is not allowlisted")
        return spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def explicit_state_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only KEV's explicit planning state, ignoring all other fields."""

    if not isinstance(state, Mapping):
        raise TypeError("explicit state must be a mapping, never raw language")
    snapshot: dict[str, Any] = {}
    for key in EXPLICIT_STATE_KEYS:
        default: Any = {} if key == "facts" else []
        snapshot[key] = _json_clone(state.get(key, default), path=f"$.{key}")
    return snapshot


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """A typed tool request derived from one explicit GOAL frame."""

    tool: str
    arguments: Any
    state_hash: str
    goal_id: str | None = None
    plan_hash: str = ""

    def __post_init__(self) -> None:
        arguments = _json_clone(self.arguments, path="$.arguments")
        object.__setattr__(self, "arguments", arguments)
        body = {
            "schema": "kev.action-plan.v1",
            "tool": self.tool,
            "arguments": arguments,
            "state_hash": self.state_hash,
            "goal_id": self.goal_id,
        }
        expected = canonical_hash(body)
        if self.plan_hash and self.plan_hash != expected:
            raise ValueError("plan_hash does not match plan contents")
        object.__setattr__(self, "plan_hash", expected)


class StatePlanner:
    """Derive bounded actions solely from structured GOAL entries.

    A goal can contain either ``{"action": {"tool": ..., "input": ...}}`` or
    the same ``tool``/``input`` fields directly.  Free-form goal text is never
    parsed into an action.  Unknown, mutating, or schema-invalid requests are
    omitted rather than being guessed into a hidden command language.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def plan(self, state: Mapping[str, Any]) -> tuple[ActionPlan, ...]:
        snapshot = explicit_state_snapshot(state)
        state_hash = canonical_hash(snapshot)
        raw_goals = snapshot["goals"]
        goals: Sequence[Any]
        if isinstance(raw_goals, Mapping):
            goals = tuple(raw_goals.values())
        elif isinstance(raw_goals, list):
            goals = raw_goals
        else:
            return ()

        plans: list[ActionPlan] = []
        for goal in goals:
            if not isinstance(goal, Mapping):
                continue
            request = goal.get("action", goal)
            if not isinstance(request, Mapping):
                continue
            tool_name = request.get("tool")
            if not isinstance(tool_name, str):
                continue
            spec = self.registry.get(tool_name)
            if spec is None or spec.production_mutation:
                continue
            arguments = request.get("input", request.get("arguments", {}))
            try:
                validate_schema(arguments, spec.input_schema, "$.input")
            except SchemaValidationError:
                continue
            goal_id = goal.get("id")
            plans.append(
                ActionPlan(
                    tool=tool_name,
                    arguments=arguments,
                    state_hash=state_hash,
                    goal_id=str(goal_id) if goal_id is not None else None,
                )
            )
        return tuple(plans)


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """An action receipt and, only on verified success, its observation."""

    receipt: Mapping[str, Any]
    observation: Mapping[str, Any] | None

    @property
    def succeeded(self) -> bool:
        return self.receipt.get("status") == "SUCCEEDED"


class ActionExecutor:
    """Execute allowlisted tools and persist a receipt for every attempt."""

    def __init__(
        self,
        registry: ToolRegistry,
        ledger_sink: LedgerSink,
        *,
        clock: Callable[[], datetime | str] | None = None,
    ) -> None:
        if not callable(ledger_sink):
            raise TypeError("ledger_sink must be callable")
        self.registry = registry
        self.ledger_sink = ledger_sink
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self) -> str:
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return str(value)

    def execute(
        self,
        action: ActionPlan | str,
        inputs: Any = None,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ActionOutcome:
        """Execute a plan or direct typed request.

        Direct execution is useful for a caller that already selected a tool;
        passing ``state`` still records only a hash of the explicit state.  The
        method returns failures as evidence rather than discarding or raising
        them.  The sole exception is :class:`ReceiptWriteError`, because a
        successful tool result must never be called VERIFIED without a durable
        receipt.
        """

        started_at = self._now()
        if isinstance(action, ActionPlan):
            tool_name = action.tool
            supplied_input = action.arguments
            state_hash = action.state_hash
            plan_hash = action.plan_hash
        else:
            tool_name = action
            supplied_input = {} if inputs is None else inputs
            try:
                snapshot = explicit_state_snapshot(state or {})
                state_hash = canonical_hash(snapshot)
            except Exception:
                state_hash = None
            plan_hash = None

        raw_output: Any = None
        output: Any = None
        failure: dict[str, Any] | None = None
        status = "REJECTED"
        production_mutation_requested = False
        try:
            if not isinstance(tool_name, str):
                raise TypeError("tool name must be a string")
            spec = self.registry.require(tool_name)
            production_mutation_requested = spec.production_mutation
            if spec.production_mutation:
                raise PermissionError("production mutation is forbidden")
            validate_schema(supplied_input, spec.input_schema, "$.input")
            safe_input = _json_clone(supplied_input, path="$.input")
            raw_output = spec.handler(copy.deepcopy(safe_input))
            validate_schema(raw_output, spec.output_schema, "$.output")
            output = _json_clone(raw_output, path="$.output")
            status = "SUCCEEDED"
        except Exception as exc:  # Preserve policy, validation, and tool errors.
            if isinstance(
                exc, (KeyError, TypeError, SchemaValidationError, PermissionError)
            ):
                status = "REJECTED"
            else:
                status = "FAILED"
            code = (
                "PRODUCTION_MUTATION_FORBIDDEN"
                if isinstance(exc, PermissionError)
                else type(exc).__name__
            )
            failure = {
                "code": code,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

        finished_at = self._now()
        body: dict[str, Any] = {
            "schema": "kev.action-receipt.v1",
            "kind": "ACTION_RECEIPT",
            "tool": _receipt_safe(tool_name),
            "input": _receipt_safe(supplied_input),
            "output": output if status == "SUCCEEDED" else None,
            "raw_output": _receipt_safe(raw_output) if raw_output is not None else None,
            "status": status,
            "failure": failure,
            "production_mutation_requested": production_mutation_requested,
            "production_mutation_performed": False,
            "state_hash": state_hash,
            "plan_hash": plan_hash,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        receipt = {**body, "receipt_hash": canonical_hash(body)}

        try:
            # The sink receives its own copy so a mutable sink cannot alter the
            # receipt that backs the returned observation.
            self.ledger_sink(copy.deepcopy(receipt))
        except Exception as exc:
            raise ReceiptWriteError(receipt, exc) from exc

        observation: dict[str, Any] | None = None
        if status == "SUCCEEDED":
            observation_body = {
                "schema": "kev.frame.v1",
                "kind": "OBSERVATION",
                "relation": "TOOL_RESULT",
                "slots": {"result": {"type": "json", "value": output}},
                "value": output,
                "status": "VERIFIED",
                "source": {"type": "TOOL", "tool": tool_name},
                "provenance": {
                    "source_type": "TOOL",
                    "source_id": tool_name,
                    "receipt_hash": receipt["receipt_hash"],
                },
                "tool": tool_name,
                "receipt_hash": receipt["receipt_hash"],
                "observed_at": finished_at,
            }
            observation = {
                **observation_body,
                "id": "observation-"
                + canonical_hash(observation_body).split(":", 1)[1][:20],
            }
        return ActionOutcome(receipt=receipt, observation=observation)


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    """Verify the self-hash of an action receipt without changing it."""

    if not isinstance(receipt, Mapping):
        return False
    candidate = dict(receipt)
    claimed = candidate.pop("receipt_hash", None)
    try:
        return isinstance(claimed, str) and claimed == canonical_hash(candidate)
    except (TypeError, ValueError):
        return False


def _frame_slots(frame: Mapping[str, Any]) -> dict[str, Any]:
    slots = frame.get("slots", {})
    merged = {}
    if isinstance(slots, Mapping):
        for key, value in slots.items():
            if isinstance(value, Mapping) and "value" in value and "type" in value:
                merged[key] = value["value"]
            else:
                merged[key] = value
    for key in (
        "subject",
        "metric",
        "key",
        "operator",
        "expected",
        "target",
        "value",
        "actual",
        "tolerance",
    ):
        if key in frame:
            merged[key] = frame[key]
    return merged


def _timestamp(frame: Mapping[str, Any]) -> datetime | None:
    for key in ("observed_at", "predicted_at", "created_at", "time"):
        value = frame.get(key)
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _comparison(
    operator: str, actual: Any, expected: Any, tolerance: float | None
) -> bool:
    aliases = {
        "=": "==",
        "eq": "==",
        "ne": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
    }
    operator = aliases.get(operator.casefold(), operator)
    if operator == "==":
        if (
            tolerance is not None
            and isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
        ):
            return math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0)
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    raise ValueError(f"unsupported prediction operator {operator!r}")


def score_predictions(
    predictions: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score structured predictions against matching later observations.

    Matching uses ``prediction_id`` when present, otherwise the exact
    ``subject``/``metric``/``key`` slot.  If timestamps are present, only an
    observation strictly later than the prediction is eligible.  With no
    timestamps, callers assert temporal order by passing the later observation
    collection to this function.
    """

    observation_list = [
        dict(item) for item in observations if isinstance(item, Mapping)
    ]
    scored: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            continue
        pred = dict(prediction)
        slots = _frame_slots(pred)
        subject = slots.get("subject", slots.get("metric", slots.get("key")))
        expected = slots.get("expected", slots.get("target", slots.get("value")))
        operator = str(slots.get("operator", "=="))
        tolerance_value = slots.get("tolerance")
        tolerance = (
            float(tolerance_value)
            if isinstance(tolerance_value, (int, float))
            else None
        )
        pred_id = pred.get("id")
        pred_time = _timestamp(pred)

        matches: list[Mapping[str, Any]] = []
        for observation in observation_list:
            obs_slots = _frame_slots(observation)
            explicit_link = observation.get("prediction_id")
            if explicit_link is not None:
                same_target = pred_id is not None and explicit_link == pred_id
            else:
                obs_subject = obs_slots.get(
                    "subject", obs_slots.get("metric", obs_slots.get("key"))
                )
                if (
                    obs_subject is None
                    and isinstance(subject, str)
                    and isinstance(observation.get("value"), Mapping)
                ):
                    if subject in observation["value"]:
                        obs_subject = subject
                same_target = subject is not None and obs_subject == subject
            if not same_target:
                continue
            obs_time = _timestamp(observation)
            if pred_time is not None and obs_time is not None and obs_time <= pred_time:
                continue
            matches.append(observation)

        if matches and any(_timestamp(item) is not None for item in matches):
            matches.sort(
                key=lambda item: (
                    _timestamp(item) or datetime.max.replace(tzinfo=timezone.utc)
                )
            )
        match = matches[0] if matches else None
        base = {
            "schema": "kev.prediction-score.v1",
            "prediction_id": pred_id if pred_id is not None else f"prediction-{index}",
            "subject": subject,
            "operator": operator,
            "expected": expected,
        }
        if match is None:
            scored.append(
                {**base, "status": "UNRESOLVED", "score": None, "observation": None}
            )
            continue

        actual_slots = _frame_slots(match)
        actual = actual_slots.get("actual", actual_slots.get("value"))
        if isinstance(actual, Mapping) and subject in actual:
            actual = actual[subject]
        elif (
            actual is None
            and isinstance(match.get("value"), Mapping)
            and subject in match["value"]
        ):
            actual = match["value"][subject]
        try:
            confirmed = _comparison(operator, actual, expected, tolerance)
            scored.append(
                {
                    **base,
                    "actual": actual,
                    "status": "CONFIRMED" if confirmed else "REFUTED",
                    "score": 1.0 if confirmed else 0.0,
                    "observation": {
                        "id": match.get("id"),
                        "receipt_hash": match.get("receipt_hash"),
                    },
                }
            )
        except (TypeError, ValueError) as exc:
            scored.append(
                {
                    **base,
                    "actual": actual,
                    "status": "UNRESOLVED",
                    "score": None,
                    "observation": {
                        "id": match.get("id"),
                        "receipt_hash": match.get("receipt_hash"),
                    },
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    return scored


class StateNarrator:
    """A non-writing language surface over supplied state and receipts."""

    def narrate(
        self,
        state: Mapping[str, Any],
        receipts: Iterable[Mapping[str, Any]] = (),
        *,
        generated_text: str | None = None,
    ) -> dict[str, Any]:
        snapshot = explicit_state_snapshot(state)
        receipt_copies = [_json_clone(item, path="$.receipts[]") for item in receipts]
        source_hashes = [
            item["receipt_hash"]
            for item in receipt_copies
            if isinstance(item, Mapping) and isinstance(item.get("receipt_hash"), str)
        ]
        if generated_text is None:
            text = "STATE " + canonical_json(snapshot)
            if receipt_copies:
                text += " RECEIPTS " + canonical_json(receipt_copies)
            grounding = "GROUNDED"
        else:
            # Arbitrary generated prose is never silently promoted to evidence.
            text = str(generated_text)
            grounding = "UNGROUNDED"
        return {
            "schema": "kev.narration.v1",
            "text": text,
            "grounding": grounding,
            "state_hash": canonical_hash(snapshot),
            "receipt_hashes": source_hashes,
            "fact_updates": [],
        }


def narrate_state(
    state: Mapping[str, Any], receipts: Iterable[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Convenience wrapper for deterministic, grounded state narration."""

    return StateNarrator().narrate(state, receipts)
