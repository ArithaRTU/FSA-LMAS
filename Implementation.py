from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from neo4j import GraphDatabase
from openai import OpenAI



# USER-DEFINED EXECUTION VARIABLES
USER_QUERY = (
    ""
)

PROPAGATION_THRESHOLD = 0.70



# CONFIGURATION
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fsa_lmas")

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-5.2")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2")
GENERATOR_MODEL_DEFAULT = os.getenv("GENERATOR_MODEL_DEFAULT", "gpt-5.2")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-5.2")
CHANGE_MANAGER_MODEL = os.getenv("CHANGE_MANAGER_MODEL", "gpt-5.2")

PROMPT_LOG_ROOT = os.getenv("PROMPT_LOG_ROOT", "./prompt_logs")

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_PROPERTY_VALUE_TYPES = (str, int, float, bool, list, dict, type(None))

CANONICAL_GENERATOR_OUTPUT_FORMAT = "canonical_graph_mutations_v1"
CANONICAL_GENERATOR_OUTPUT_CONTRACT = """
The Generator output wire format is fixed system-wide as canonical_graph_mutations_v1.

Return exactly one JSON object with exactly these top-level keys:
{
  "answer_text": "string",
  "graph_mutations": {
    "updates": [
      {
        "target": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        },
        "properties": {
          "SomeEditableProperty": "value"
        }
      }
    ],
    "create_nodes": [
      {
        "label": "string",
        "properties": {
          "id": "unique-id-or-key",
          "SomeProperty": "value"
        }
      }
    ],
    "create_relationships": [
      {
        "source": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        },
        "rel_type": "string",
        "target": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        }
      }
    ],
    "delete_nodes": [
      {
        "label": "string",
        "key_name": "string",
        "key_value": "string"
      }
    ]
  }
}

Do not rename keys.
Do not emit top-level mutation keys outside graph_mutations.
Do not use aliases such as mutations, add_nodes, add_edges, add_relationships, from/to/type, match, match_by, match_props, key, relationship_type, ref, or temp_id.
If you create a node and then connect to it, reference it in create_relationships by the same label/key_name/key_value that already appears in create_nodes.
If no graph changes are needed, return empty arrays for all graph_mutations fields.
answer_text must always be present, even if brief.
""".strip()
CANONICAL_GENERATOR_JSON_SCHEMA = """
{
  "answer_text": "string",
  "graph_mutations": {
    "updates": [
      {
        "target": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        },
        "properties": {
          "SomeEditableProperty": "value"
        }
      }
    ],
    "create_nodes": [
      {
        "label": "string",
        "properties": {
          "id": "unique-id-or-key",
          "SomeProperty": "value"
        }
      }
    ],
    "create_relationships": [
      {
        "source": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        },
        "rel_type": "string",
        "target": {
          "label": "string",
          "key_name": "string",
          "key_value": "string"
        }
      }
    ],
    "delete_nodes": [
      {
        "label": "string",
        "key_name": "string",
        "key_value": "string"
      }
    ]
  }
}
""".strip()



# TYPES
Level = Literal["All", "StudyField", "Program", "Course", "Topic"]


@dataclass
class CandidateNode:
    label: str
    key_name: str
    key_value: str
    display_name: str
    score: float
    summary: str = ""


@dataclass
class RoutingDecision:
    scope_level: Level
    specific: bool
    applies_to_entire_scope: bool
    target_label: Optional[str]
    target_key_name: Optional[str]
    target_key_value: Optional[str]
    rationale: str


@dataclass
class OrchestrationPlan:
    model: str
    profile_prompt: str
    evaluation_criteria: List[str]
    output_format: str
    goal: str
    confidence_focus: str


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    strengths: List[str]
    weaknesses: List[str]
    improvement_actions: List[str]
    rationale: str


@dataclass
class GraphMutationPlan:
    updates: List[Dict[str, Any]]
    create_nodes: List[Dict[str, Any]]
    create_relationships: List[Dict[str, Any]]
    delete_nodes: List[Dict[str, Any]]


@dataclass
class GeneratorResult:
    answer_text: str
    graph_mutations: GraphMutationPlan


@dataclass
class BFUExecution:
    execution_id: str
    level: str
    node_label: str
    node_key_name: str
    node_key_value: str
    node_name: str
    user_query: str
    plan: OrchestrationPlan
    generator_output: str
    graph_mutations: Dict[str, Any]
    evaluation: EvaluationResult
    mutation_results: Dict[str, Any] = field(default_factory=dict)
    propagated: bool = False
    propagation_source_execution_id: Optional[str] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PropagationRecord:
    execution_id: str
    node_label: str
    node_key_name: str
    node_key_value: str
    node_name: str
    level: str
    user_query: str
    propagated: bool
    propagation_source_execution_id: Optional[str]
    timestamp_utc: str
    model: str
    profile_prompt: str
    evaluation_criteria: List[str]
    output_format: str
    goal: str
    confidence_focus: str
    score: float
    passed: bool
    strengths: List[str]
    weaknesses: List[str]
    improvement_actions: List[str]
    rationale: str



# HELPERS
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def truncate(text: Optional[str], limit: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def _is_scalar_property_value(value: Any) -> bool:
    return value not in (None, "") and isinstance(value, (str, int, float, bool))


def _extract_single_match_ref(
    ref: Any,
    *,
    default_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(ref, dict):
        return None

    ref_id = ref.get("temp_id") or ref.get("ref")
    if ref_id not in (None, ""):
        return {"temp_id": str(ref_id)}

    label = ref.get("label") or ref.get("type") or default_label
    match = ref.get("match")
    if not isinstance(match, dict):
        match = ref.get("match_by")
    if not isinstance(match, dict):
        match = ref.get("match_props")
    if not isinstance(match, dict):
        match = ref.get("key")
    if not isinstance(match, dict):
        match = ref.get("properties")
    if not isinstance(match, dict):
        key_name = ref.get("key_name")
        key_value = ref.get("key_value")
        if key_name not in (None, "") and _is_scalar_property_value(key_value):
            return {
                "label": label,
                "key_name": key_name,
                "key_value": key_value,
            }
        return None

    scalar_items = [
        (str(key), value)
        for key, value in match.items()
        if _is_scalar_property_value(value)
    ]
    if not scalar_items:
        return None

    key_name, key_value = scalar_items[0]
    return {
        "label": label,
        "key_name": key_name,
        "key_value": key_value,
    }


def _canonicalize_create_node(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = item.get("label") or item.get("type")
    properties = item.get("properties")
    if label in (None, "") or not isinstance(properties, dict):
        return None

    payload: Dict[str, Any] = {
        "label": label,
        "properties": properties,
    }
    if item.get("temp_id") not in (None, ""):
        payload["temp_id"] = str(item["temp_id"])
    if item.get("ref") not in (None, ""):
        payload["ref"] = str(item["ref"])
    return payload


def _canonicalize_delete_node(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = item.get("label") or item.get("type")
    key_name = item.get("key_name")
    key_value = item.get("key_value")
    if label in (None, "") or key_name in (None, "") or key_value in (None, ""):
        return None
    return {
        "label": label,
        "key_name": key_name,
        "key_value": key_value,
    }


def _canonicalize_update_node(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    properties = item.get("properties")
    if not isinstance(properties, dict):
        return None

    target = item.get("target")
    if not isinstance(target, dict):
        return None

    target_ref = _extract_single_match_ref(target)
    if target_ref is None or "temp_id" in target_ref:
        return None

    return {
        "target": target_ref,
        "properties": properties,
    }


def _canonicalize_create_relationship(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_raw = item.get("source")
    if not isinstance(source_raw, dict):
        source_raw = item.get("from")
    target_raw = item.get("target")
    if not isinstance(target_raw, dict):
        target_raw = item.get("to")
    rel_type = item.get("rel_type") or item.get("type") or item.get("relationship_type")

    if not isinstance(source_raw, dict) or not isinstance(target_raw, dict) or rel_type in (None, ""):
        return None

    source_ref = _extract_single_match_ref(source_raw)
    target_ref = _extract_single_match_ref(target_raw)
    if source_ref is None or target_ref is None:
        return None

    return {
        "source": source_ref,
        "target": target_ref,
        "rel_type": rel_type,
    }


def _canonicalize_mutation_item(item: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(item, dict):
        return None

    relationship = _canonicalize_create_relationship(item)
    if relationship is not None:
        return "create_relationships", relationship

    create_node = _canonicalize_create_node(item)
    if create_node is not None:
        return "create_nodes", create_node

    delete_node = _canonicalize_delete_node(item)
    if delete_node is not None:
        return "delete_nodes", delete_node

    update_node = _canonicalize_update_node(item)
    if update_node is not None:
        return "updates", update_node

    action = str(item.get("action") or item.get("op") or "").strip().lower()
    if action in {"create_node", "add_node"} and isinstance(item.get("node"), dict):
        nested = dict(item["node"])
        if item.get("temp_id") not in (None, "") and nested.get("temp_id") in (None, ""):
            nested["temp_id"] = item["temp_id"]
        if item.get("ref") not in (None, "") and nested.get("ref") in (None, ""):
            nested["ref"] = item["ref"]
        return _canonicalize_mutation_item(nested)

    if action == "delete_node" and isinstance(item.get("node"), dict):
        return _canonicalize_mutation_item(item["node"])

    if action == "update_node":
        update_payload = {
            "target": item.get("target"),
            "properties": item.get("properties"),
        }
        update_node = _canonicalize_update_node(update_payload)
        if update_node is not None:
            return "updates", update_node

    if action in {"create_relationship", "add_relationship"}:
        relationship_payload = item.get("relationship")
        if not isinstance(relationship_payload, dict):
            relationship_payload = item
        relationship = _canonicalize_create_relationship(relationship_payload)
        if relationship is not None:
            return "create_relationships", relationship

    return None


def extract_graph_mutations_payload(raw_output: Any) -> Any:
    if not isinstance(raw_output, dict):
        return {}

    direct = raw_output.get("graph_mutations")
    if isinstance(direct, (dict, list)) and direct:
        return direct

    mutation_keys = {
        "updates",
        "create_nodes",
        "create_relationships",
        "delete_nodes",
        "add_nodes",
        "add_edges",
        "add_relationships",
        "mutations",
        "node_updates",
    }
    if any(key in raw_output for key in mutation_keys):
        return raw_output

    return {}


def normalize_graph_mutations(raw_mutations: Any) -> Dict[str, List[Dict[str, Any]]]:
    normalized: Dict[str, List[Dict[str, Any]]] = {
        "updates": [],
        "create_nodes": [],
        "create_relationships": [],
        "delete_nodes": [],
    }

    if not raw_mutations:
        return normalized

    if isinstance(raw_mutations, dict):
        alias_map = {
            "add_nodes": "create_nodes",
            "nodes_to_create": "create_nodes",
            "new_nodes": "create_nodes",
            "add_edges": "create_relationships",
            "add_relationships": "create_relationships",
            "edges_to_create": "create_relationships",
            "new_relationships": "create_relationships",
            "relationships": "create_relationships",
            "links": "create_relationships",
            "node_updates": "updates",
            "mutations": "__mutation_items__",
        }

        for key in normalized:
            value = raw_mutations.get(key, [])
            if isinstance(value, list):
                for item in value:
                    canonical = _canonicalize_mutation_item(item) if isinstance(item, dict) else None
                    if canonical is None:
                        continue
                    bucket, payload = canonical
                    normalized[bucket].append(payload)

        for source_key, target_key in alias_map.items():
            value = raw_mutations.get(source_key)
            if not isinstance(value, list):
                continue
            if target_key == "__mutation_items__":
                for item in value:
                    canonical = _canonicalize_mutation_item(item)
                    if canonical is None:
                        continue
                    bucket, payload = canonical
                    normalized[bucket].append(payload)
            else:
                for item in value:
                    canonical = _canonicalize_mutation_item(item) if isinstance(item, dict) else None
                    if canonical is None:
                        continue
                    bucket, payload = canonical
                    normalized[bucket].append(payload)

        single_item = _canonicalize_mutation_item(raw_mutations)
        if single_item is not None:
            bucket, payload = single_item
            normalized[bucket].append(payload)

        if any(normalized.values()):
            logger.info(
                "Normalized graph_mutations buckets: %s",
                {key: len(value) for key, value in normalized.items()},
            )
        return normalized

    if isinstance(raw_mutations, list):
        for item in raw_mutations:
            canonical = _canonicalize_mutation_item(item) if isinstance(item, dict) else None
            if canonical is None:
                continue
            bucket, payload = canonical
            normalized[bucket].append(payload)

        logger.warning(
            "Normalized list-based graph_mutations into structured buckets: %s",
            {key: len(value) for key, value in normalized.items()},
        )
        return normalized

    logger.warning("Ignoring unsupported graph_mutations payload of type %s", type(raw_mutations).__name__)
    return normalized


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def map_label_to_level(label: str) -> str:
    return {
        "RTUStudyField": "StudyField",
        "RTUProgram": "Program",
        "RTUCourse": "Course",
        "RTUTopic": "Topic",
    }.get(label, label)


def map_level_to_label(level: str) -> str:
    return {
        "StudyField": "RTUStudyField",
        "Program": "RTUProgram",
        "Course": "RTUCourse",
        "Topic": "RTUTopic",
    }.get(level, level)


def coerce_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def select_identity_property(properties: Dict[str, Any]) -> Tuple[str, Any]:
    candidates = [
        "id",
        "ID",
        "uuid",
        "UUID",
        "url",
        "URL",
        "source_url",
        "sourceUrl",
        "citation",
        "title",
        "name",
        "label",
    ]
    for key in candidates:
        value = properties.get(key)
        if value not in (None, ""):
            return key, value

    for key, value in properties.items():
        if value not in (None, "") and isinstance(value, (str, int, float, bool)):
            return str(key), value

    raise ValueError("Created node requires at least one non-empty scalar property to act as an identity.")


def validate_identifier(name: str, kind: str) -> str:
    if not isinstance(name, str) or not IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


def sanitize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}
    return str(value)


def deep_filter_original_props(props: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    filtered = {}
    blocked = []

    for k, v in (props or {}).items():
        key = validate_identifier(str(k), "property key")
        if key.startswith("Original"):
            blocked.append(key)
        else:
            filtered[key] = sanitize_value(v)

    return filtered, blocked


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def detect_rule_based_routing(user_query: str) -> Optional[RoutingDecision]:
    q = normalize_text(user_query)

    asks_all = contains_any(
        q,
        [
            "all",
            "across the curriculum",
            "entire curriculum",
            "whole curriculum",
            "curriculum-wide",
            "curriculum wide",
            "every",
            "for each",
            "each",
        ],
    )

    course_scope = contains_any(
        q,
        [
            "course",
            "courses",
            "course annotation",
            "course annotations",
            "syllabus",
            "syllabi",
        ],
    )

    program_scope = contains_any(
        q,
        [
            "program",
            "programs",
            "programme",
            "programmes",
        ],
    )

    topic_scope = contains_any(
        q,
        [
            "topic",
            "topics",
        ],
    )

    field_scope = contains_any(
        q,
        [
            "study field",
            "study fields",
            "field of study",
            "fields of study",
        ],
    )

    mass_update = contains_any(
        q,
        [
            "revise",
            "update",
            "augment",
            "improve",
            "rewrite",
            "assess",
            "evaluate",
            "align",
            "add recommended research articles",
            "add references",
            "add articles",
            "define one or more research articles",
            "research articles",
            "must read",
        ],
    )

    if asks_all and course_scope and mass_update:
        return RoutingDecision(
            scope_level="Course",
            specific=False,
            applies_to_entire_scope=True,
            target_label=None,
            target_key_name=None,
            target_key_value=None,
            rationale="Rule-based routing: broad curriculum-wide course request detected.",
        )

    if asks_all and program_scope and mass_update:
        return RoutingDecision(
            scope_level="Program",
            specific=False,
            applies_to_entire_scope=True,
            target_label=None,
            target_key_name=None,
            target_key_value=None,
            rationale="Rule-based routing: broad program-wide request detected.",
        )

    if asks_all and topic_scope and mass_update:
        return RoutingDecision(
            scope_level="Topic",
            specific=False,
            applies_to_entire_scope=True,
            target_label=None,
            target_key_name=None,
            target_key_value=None,
            rationale="Rule-based routing: broad topic-wide request detected.",
        )

    if asks_all and field_scope and mass_update:
        return RoutingDecision(
            scope_level="StudyField",
            specific=False,
            applies_to_entire_scope=True,
            target_label=None,
            target_key_name=None,
            target_key_value=None,
            rationale="Rule-based routing: broad study-field-wide request detected.",
        )

    if asks_all and mass_update and contains_any(
        q,
        [
            "curriculum",
            "curriculum-wide",
            "curriculum wide",
            "across the curriculum",
            "entire curriculum",
            "whole curriculum",
        ],
    ):
        return RoutingDecision(
            scope_level="All",
            specific=False,
            applies_to_entire_scope=True,
            target_label=None,
            target_key_name=None,
            target_key_value=None,
            rationale="Rule-based routing: broad curriculum-wide multi-level request detected.",
        )

    return None



# PROMPT ARCHIVE
class PromptArchive:
    def __init__(self, root: str = PROMPT_LOG_ROOT, run_id: Optional[str] = None) -> None:
        self.root = Path(root)
        ensure_dir(self.root)
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.run_dir = self.root / self.run_id
        ensure_dir(self.run_dir)

    def manager_dir(self, manager_name: str, execution_id: Optional[str] = None) -> Path:
        suffix = execution_id or uuid.uuid4().hex[:8]
        path = self.run_dir / f"{slugify(manager_name)}_{suffix}"
        ensure_dir(path)
        return path

    def bfu_dir(self, node_name: str, execution_id: str) -> Path:
        path = self.run_dir / f"bfu_{slugify(node_name)}_{execution_id}"
        ensure_dir(path)
        return path

    def save_interaction(
        self,
        base_dir: Path,
        component_name: str,
        *,
        instructions: str,
        input_payload: Any,
        output_payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        cdir = base_dir / slugify(component_name)
        ensure_dir(cdir)

        (cdir / "instructions.txt").write_text(instructions or "", encoding="utf-8")

        if isinstance(input_payload, (dict, list)):
            (cdir / "input.json").write_text(
                json.dumps(input_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            (cdir / "input.txt").write_text(str(input_payload), encoding="utf-8")

        if isinstance(output_payload, (dict, list)):
            (cdir / "output.json").write_text(
                json.dumps(output_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            (cdir / "output.txt").write_text(str(output_payload), encoding="utf-8")

        if metadata is not None:
            (cdir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


class RunStateStore:
    def __init__(self, root: str = PROMPT_LOG_ROOT) -> None:
        self.root = Path(root)
        ensure_dir(self.root)
        self.state_path = self.root / "latest_run_state.json"

    def load_previous_created_nodes(self) -> List[Dict[str, Any]]:
        if not self.state_path.exists():
            return []
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse run state file: %s", self.state_path)
            return []

        created_nodes = raw.get("created_nodes", [])
        if not isinstance(created_nodes, list):
            return []
        return [node for node in created_nodes if isinstance(node, dict)]

    def save_current_run(
        self,
        *,
        run_id: str,
        run_dir: str,
        created_nodes: List[Dict[str, Any]],
    ) -> None:
        payload = {
            "run_id": run_id,
            "run_dir": run_dir,
            "saved_at_utc": utc_now_iso(),
            "created_nodes": created_nodes,
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )



# OPENAI WRAPPER
class OpenAIService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content", [])
            for piece in content or []:
                piece_text = getattr(piece, "text", None)
                if piece_text is None and isinstance(piece, dict):
                    piece_text = piece.get("text")
                if isinstance(piece_text, str):
                    parts.append(piece_text)
        return "\n".join(p for p in parts if p).strip()

    def text_response(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int = 2500,
    ) -> str:
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=max_output_tokens,
        )
        return self._extract_text(response)

    def json_response(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        json_schema_description: str,
        max_output_tokens: int = 2500,
        retries: int = 2,
    ) -> Dict[str, Any]:
        final_instructions = f"""
{instructions}

Return ONLY valid JSON.
Do not include markdown fences.
Do not add explanatory text outside the JSON.

Expected JSON contract:
{json_schema_description}
""".strip()

        last_error = None
        last_raw = None

        for _ in range(retries + 1):
            raw = self.text_response(
                model=model,
                instructions=final_instructions,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
            )
            last_raw = raw
            try:
                return safe_json_loads(raw)
            except Exception as exc:
                last_error = exc

        raise ValueError(
            f"Could not parse model JSON output. Error={last_error}; Raw={truncate(last_raw or '', 1000)}"
        )



# NEO4J REPOSITORY
class Neo4jRepository:
    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not user or not password:
            raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set.")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **params) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_node_context(self, label: str, key_name: str, key_value: str) -> Dict[str, Any]:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        q = f"""
        MATCH (n:{label})
        WHERE n.{key_name} = $key_value
        RETURN properties(n) AS props
        LIMIT 1
        """
        rows = self._run(q, key_value=key_value)
        if not rows:
            raise ValueError(f"Node not found: {label}({key_name}={key_value})")
        return rows[0]["props"]

    def get_display_name(self, label: str, props: Dict[str, Any]) -> str:
        if label == "RTUStudyField":
            return props.get("RTUStudyFieldLabel", "Unknown Study Field")
        if label == "RTUProgram":
            return props.get("RTUProgramTitle") or props.get("RTUProgramID", "Unknown Program")
        if label == "RTUCourse":
            return props.get("RTUCourseTitle") or props.get("RTUCourseCode", "Unknown Course")
        if label == "RTUTopic":
            return props.get("RTUTopicLabel", "Unknown Topic")
        return (
            props.get("name")
            or props.get("title")
            or props.get("label")
            or props.get("id")
            or label
        )

    def get_parent_chain(self, label: str, key_name: str, key_value: str) -> Dict[str, Any]:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        if label == "RTUTopic":
            q = f"""
            MATCH (t:RTUTopic)<-[:composedOf]-(c:RTUCourse)<-[:composedOf]-(p:RTUProgram)<-[:composedOf]-(f:RTUStudyField)
            WHERE t.{key_name} = $key_value
            RETURN properties(f) AS field_props, properties(p) AS program_props, properties(c) AS course_props, properties(t) AS topic_props
            LIMIT 1
            """
        elif label == "RTUCourse":
            q = f"""
            MATCH (c:RTUCourse)<-[:composedOf]-(p:RTUProgram)<-[:composedOf]-(f:RTUStudyField)
            WHERE c.{key_name} = $key_value
            RETURN properties(f) AS field_props, properties(p) AS program_props, properties(c) AS course_props
            LIMIT 1
            """
        elif label == "RTUProgram":
            q = f"""
            MATCH (p:RTUProgram)<-[:composedOf]-(f:RTUStudyField)
            WHERE p.{key_name} = $key_value
            RETURN properties(f) AS field_props, properties(p) AS program_props
            LIMIT 1
            """
        elif label == "RTUStudyField":
            q = f"""
            MATCH (f:RTUStudyField)
            WHERE f.{key_name} = $key_value
            RETURN properties(f) AS field_props
            LIMIT 1
            """
        else:
            return {
                "selected_node": self.get_node_context(label, key_name, key_value)
            }

        rows = self._run(q, key_value=key_value)
        return rows[0] if rows else {"selected_node": self.get_node_context(label, key_name, key_value)}

    def append_agent_log(self, label: str, key_name: str, key_value: str, entry: Dict[str, Any]) -> None:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        entry_json = json.dumps(entry, ensure_ascii=False)
        q = f"""
        MATCH (n:{label})
        WHERE n.{key_name} = $key_value
        SET n.AgentLog = coalesce(
            CASE
                WHEN n.AgentLog = '' THEN NULL
                ELSE n.AgentLog
            END,
            []
        ) + [$entry_json]
        """
        self._run(q, key_value=key_value, entry_json=entry_json)

    @staticmethod
    def _parse_agent_log_entries(
        raw_entries: Any,
        *,
        default_node_label: Optional[str] = None,
        default_key_name: Optional[str] = None,
        default_key_value: Optional[str] = None,
        default_node_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []

        if raw_entries in (None, ""):
            return parsed

        if isinstance(raw_entries, str):
            candidate = raw_entries.strip()
            if not candidate:
                return parsed
            try:
                raw_entries = safe_json_loads(candidate)
            except Exception:
                logger.warning("Skipping malformed AgentLog payload")
                return parsed

        if isinstance(raw_entries, dict):
            raw_entries = [raw_entries]

        if not isinstance(raw_entries, list):
            return parsed

        for entry in raw_entries:
            try:
                if isinstance(entry, str):
                    if not entry.strip():
                        continue
                    record = safe_json_loads(entry)
                elif isinstance(entry, dict):
                    record = entry
                else:
                    continue

                if not isinstance(record, dict):
                    continue

                record = dict(record)
                if default_node_label and not record.get("node_label"):
                    record["node_label"] = default_node_label
                if default_key_name and not record.get("node_key_name"):
                    record["node_key_name"] = default_key_name
                if default_key_value is not None and not record.get("node_key_value"):
                    record["node_key_value"] = default_key_value
                if default_node_name and not record.get("node_name"):
                    record["node_name"] = default_node_name
                if record.get("node_label") and not record.get("level"):
                    record["level"] = map_label_to_level(str(record["node_label"]))

                parsed.append(record)
            except Exception as exc:
                logger.warning("Skipping malformed AgentLog entry: %s", exc)

        return parsed

    def get_agent_log(self, label: str, key_name: str, key_value: str) -> List[Dict[str, Any]]:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        q = f"""
        MATCH (n:{label})
        WHERE n.{key_name} = $key_value
        RETURN properties(n) AS props, coalesce(n.AgentLog, []) AS agent_log
        LIMIT 1
        """
        rows = self._run(q, key_value=key_value)
        if not rows:
            return []

        props = rows[0].get("props", {}) or {}
        node_name = self.get_display_name(label, props)
        return self._parse_agent_log_entries(
            rows[0].get("agent_log", []),
            default_node_label=label,
            default_key_name=key_name,
            default_key_value=str(key_value),
            default_node_name=node_name,
        )

    def get_agentlog_bfu_records_for_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}

        for node in nodes:
            try:
                entries = self.get_agent_log(
                    label=str(node["label"]),
                    key_name=str(node["key_name"]),
                    key_value=str(node["key_value"]),
                )
            except Exception as exc:
                logger.warning("Failed to read AgentLog for node %s: %s", node, exc)
                continue

            for entry in entries:
                if entry.get("record_type") != "BFUExecution":
                    continue
                execution_id = str(entry.get("execution_id") or "").strip()
                if not execution_id:
                    continue
                deduped.setdefault(execution_id, entry)

        return sorted(
            deduped.values(),
            key=lambda record: str(record.get("timestamp_utc", "")),
        )

    def get_agentlog_bfu_records_for_level(self, level: str) -> List[Dict[str, Any]]:
        label = validate_identifier(map_level_to_label(level), "label")
        q = f"""
        MATCH (n:{label})
        WHERE n.AgentLog IS NOT NULL
        RETURN properties(n) AS props, coalesce(n.AgentLog, []) AS agent_log
        """
        rows = self._run(q)

        deduped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            props = row.get("props", {}) or {}
            default_key_name = next(
                (
                    candidate
                    for candidate in (
                        "RTUStudyFieldLabel",
                        "RTUProgramID",
                        "RTUCourseCode",
                        "RTUTopicLabel",
                    )
                    if props.get(candidate) not in (None, "")
                ),
                None,
            )
            default_key_value = props.get(default_key_name) if default_key_name else None
            node_name = self.get_display_name(label, props)

            for entry in self._parse_agent_log_entries(
                row.get("agent_log", []),
                default_node_label=label,
                default_key_name=default_key_name,
                default_key_value=str(default_key_value) if default_key_value is not None else None,
                default_node_name=node_name,
            ):
                if entry.get("record_type") != "BFUExecution":
                    continue
                execution_id = str(entry.get("execution_id") or "").strip()
                if not execution_id:
                    continue
                deduped.setdefault(execution_id, entry)

        return sorted(
            deduped.values(),
            key=lambda record: str(record.get("timestamp_utc", "")),
        )

    def search_candidates(self, query_text: str, limit_per_label: int = 2000) -> List[CandidateNode]:
        q_text = normalize_text(query_text)

        if not q_text:
            return []

        cypher = """
        CALL {
            MATCH (f:RTUStudyField)
            WHERE toLower(coalesce(f.RTUStudyFieldLabel, "")) CONTAINS $q
               OR toLower(coalesce(f.OriginalRTUStudyField, "")) CONTAINS $q
            RETURN "RTUStudyField" AS label,
                   "RTUStudyFieldLabel" AS key_name,
                   f.RTUStudyFieldLabel AS key_value,
                   f.RTUStudyFieldLabel AS display_name,
                   coalesce(f.OriginalRTUStudyField, "") AS summary

            UNION

            MATCH (p:RTUProgram)
            WHERE toLower(coalesce(p.RTUProgramTitle, "")) CONTAINS $q
               OR toLower(coalesce(p.RTUProgramAbstract, "")) CONTAINS $q
               OR toLower(coalesce(p.RTUProgramAims, "")) CONTAINS $q
               OR toLower(coalesce(p.RTUProgramID, "")) CONTAINS $q
            RETURN "RTUProgram" AS label,
                   "RTUProgramID" AS key_name,
                   p.RTUProgramID AS key_value,
                   coalesce(p.RTUProgramTitle, p.RTUProgramID) AS display_name,
                   coalesce(p.RTUProgramAbstract, "") AS summary

            UNION

            MATCH (c:RTUCourse)
            WHERE toLower(coalesce(c.RTUCourseTitle, "")) CONTAINS $q
               OR toLower(coalesce(c.RTUCourseAnnotation, "")) CONTAINS $q
               OR toLower(coalesce(c.RTUCourseCode, "")) CONTAINS $q
               OR toLower(coalesce(c.RTUCourseFieldofStudy, "")) CONTAINS $q
            RETURN "RTUCourse" AS label,
                   "RTUCourseCode" AS key_name,
                   c.RTUCourseCode AS key_value,
                   coalesce(c.RTUCourseTitle, c.RTUCourseCode) AS display_name,
                   coalesce(c.RTUCourseAnnotation, "") AS summary

            UNION

            MATCH (t:RTUTopic)
            WHERE toLower(coalesce(t.RTUTopicLabel, "")) CONTAINS $q
               OR toLower(coalesce(t.OriginalRTUTopic, "")) CONTAINS $q
            RETURN "RTUTopic" AS label,
                   "RTUTopicLabel" AS key_name,
                   t.RTUTopicLabel AS key_value,
                   t.RTUTopicLabel AS display_name,
                   coalesce(t.OriginalRTUTopic, "") AS summary
        }
        RETURN label, key_name, key_value, display_name, summary
        """

        rows = self._run(cypher, q=q_text)

        grouped_counts: Dict[str, int] = {}
        limited_rows: List[Dict[str, Any]] = []
        for row in rows:
            lbl = row["label"]
            current = grouped_counts.get(lbl, 0)
            if current < limit_per_label:
                limited_rows.append(row)
                grouped_counts[lbl] = current + 1

        results = []
        for r in limited_rows:
            display = (r["display_name"] or "").lower()
            summary = (r.get("summary") or "").lower()
            score = 0.5
            if q_text and q_text in display:
                score += 0.35
            if q_text and q_text in summary:
                score += 0.15

            results.append(
                CandidateNode(
                    label=r["label"],
                    key_name=r["key_name"],
                    key_value=r["key_value"],
                    display_name=r["display_name"],
                    score=score,
                    summary=truncate(r.get("summary", ""), 300),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def get_nodes_by_level(self, level: Level) -> List[CandidateNode]:
        if level == "StudyField":
            q = """
            MATCH (n:RTUStudyField)
            RETURN "RTUStudyField" AS label, "RTUStudyFieldLabel" AS key_name,
                   n.RTUStudyFieldLabel AS key_value, n.RTUStudyFieldLabel AS display_name,
                   coalesce(n.OriginalRTUStudyField, "") AS summary
            """
        elif level == "Program":
            q = """
            MATCH (n:RTUProgram)
            RETURN "RTUProgram" AS label, "RTUProgramID" AS key_name,
                   n.RTUProgramID AS key_value, coalesce(n.RTUProgramTitle, n.RTUProgramID) AS display_name,
                   coalesce(n.RTUProgramAbstract, "") AS summary
            """
        elif level == "Course":
            q = """
            MATCH (n:RTUCourse)
            RETURN "RTUCourse" AS label, "RTUCourseCode" AS key_name,
                   n.RTUCourseCode AS key_value, coalesce(n.RTUCourseTitle, n.RTUCourseCode) AS display_name,
                   coalesce(n.RTUCourseAnnotation, "") AS summary
            """
        elif level == "Topic":
            q = """
            MATCH (n:RTUTopic)
            RETURN "RTUTopic" AS label, "RTUTopicLabel" AS key_name,
                   n.RTUTopicLabel AS key_value, n.RTUTopicLabel AS display_name,
                   coalesce(n.OriginalRTUTopic, "") AS summary
            """
        else:
            return []

        rows = self._run(q)
        return [
            CandidateNode(
                label=r["label"],
                key_name=r["key_name"],
                key_value=r["key_value"],
                display_name=r["display_name"],
                score=0.5,
                summary=truncate(r.get("summary", ""), 300),
            )
            for r in rows
        ]

    def get_all_curriculum_nodes(self) -> List[CandidateNode]:
        out = []
        for level in ["StudyField", "Program", "Course", "Topic"]:
            out.extend(self.get_nodes_by_level(level))
        return out

    def update_node_properties(
        self,
        label: str,
        key_name: str,
        key_value: str,
        properties: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        filtered, blocked = deep_filter_original_props(properties)
        if filtered:
            q = f"""
            MATCH (n:{label})
            WHERE n.{key_name} = $key_value
            SET n += $props
            """
            self._run(q, key_value=key_value, props=filtered)

        return filtered, blocked

    def create_node(self, label: str, properties: Dict[str, Any]) -> None:
        label = validate_identifier(label, "label")
        sanitized = {
            validate_identifier(str(k), "property key"): sanitize_value(v)
            for k, v in (properties or {}).items()
        }
        q = f"""
        CREATE (n:{label})
        SET n = $props
        """
        self._run(q, props=sanitized)

    def create_relationship(
        self,
        source_label: str,
        source_key_name: str,
        source_key_value: str,
        rel_type: str,
        target_label: str,
        target_key_name: str,
        target_key_value: str,
    ) -> None:
        source_label = validate_identifier(source_label, "label")
        source_key_name = validate_identifier(source_key_name, "property key")
        target_label = validate_identifier(target_label, "label")
        target_key_name = validate_identifier(target_key_name, "property key")
        rel_type = validate_identifier(rel_type, "relationship type")

        source_count_q = f"""
        MATCH (n:{source_label})
        WHERE n.{source_key_name} = $key_value
        RETURN count(n) AS count
        """
        source_count_rows = self._run(source_count_q, key_value=source_key_value)
        source_count = int(source_count_rows[0]["count"]) if source_count_rows else 0
        if source_count != 1:
            raise ValueError(
                f"Source reference must match exactly one node, got {source_count}: "
                f"{source_label}({source_key_name}={source_key_value})"
            )

        target_count_q = f"""
        MATCH (n:{target_label})
        WHERE n.{target_key_name} = $key_value
        RETURN count(n) AS count
        """
        target_count_rows = self._run(target_count_q, key_value=target_key_value)
        target_count = int(target_count_rows[0]["count"]) if target_count_rows else 0
        if target_count != 1:
            raise ValueError(
                f"Target reference must match exactly one node, got {target_count}: "
                f"{target_label}({target_key_name}={target_key_value})"
            )

        q = f"""
        MATCH (a:{source_label})
        WHERE a.{source_key_name} = $source_key_value
        MATCH (b:{target_label})
        WHERE b.{target_key_name} = $target_key_value
        MERGE (a)-[r:{rel_type}]->(b)
        """
        self._run(
            q,
            source_key_value=source_key_value,
            target_key_value=target_key_value,
        )

    def delete_node(self, label: str, key_name: str, key_value: Any) -> None:
        label = validate_identifier(label, "label")
        key_name = validate_identifier(key_name, "property key")

        q = f"""
        MATCH (n:{label})
        WHERE n.{key_name} = $key_value
        DETACH DELETE n
        """
        self._run(q, key_value=key_value)



# FRACTAL MANAGER
class FractalManager:
    def __init__(self, llm: OpenAIService, repo: Neo4jRepository, archive: PromptArchive) -> None:
        self.llm = llm
        self.repo = repo
        self.archive = archive

    def route_query(self, user_query: str) -> RoutingDecision:
        manager_id = uuid.uuid4().hex[:10]
        mdir = self.archive.manager_dir("fractal_manager", manager_id)

        rule_based = detect_rule_based_routing(user_query)
        if rule_based is not None:
            self.archive.save_interaction(
                mdir,
                "routing_rule_based",
                instructions="Apply rule-based routing before LLM routing.",
                input_payload={"user_query": user_query},
                output_payload=asdict(rule_based),
                metadata={"component": "fractal_manager"},
            )
            return rule_based

        candidates = self.repo.search_candidates(user_query, limit_per_label=50)

        instructions = """
You are the Fractal Manager of a hierarchical curriculum graph system.

Decide:
1. Which scope level the query targets:
   - All
   - StudyField
   - Program
   - Course
   - Topic
2. Whether it targets one specific node
3. Whether it applies to the entire chosen scope

Rules:
- Use specific=true only when one candidate is clearly intended.
- Use applies_to_entire_scope=true when the user asks to revise, update, augment, or assess all items in a level or across the curriculum.
- Use All when the query spans multiple levels or clearly concerns the whole curriculum.
- Prefer Course when the query clearly asks to revise course annotations, course descriptions, or all courses.
"""

        schema = """
{
  "scope_level": "All|StudyField|Program|Course|Topic",
  "specific": true,
  "applies_to_entire_scope": false,
  "target_label": "string or null",
  "target_key_name": "string or null",
  "target_key_value": "string or null",
  "rationale": "string"
}
""".strip()

        payload = {
            "user_query": user_query,
            "candidate_nodes": [asdict(c) for c in candidates[:50]],
            "hierarchy": "RTUStudyField -> RTUProgram -> RTUCourse -> RTUTopic",
        }

        raw = self.llm.json_response(
            model=ROUTER_MODEL,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema_description=schema,
            max_output_tokens=1200,
        )

        self.archive.save_interaction(
            mdir,
            "routing",
            instructions=instructions,
            input_payload=payload,
            output_payload=raw,
            metadata={"model": ROUTER_MODEL},
        )

        return RoutingDecision(
            scope_level=raw.get("scope_level", "All"),
            specific=bool(raw.get("specific", False)),
            applies_to_entire_scope=bool(raw.get("applies_to_entire_scope", False)),
            target_label=raw.get("target_label"),
            target_key_name=raw.get("target_key_name"),
            target_key_value=raw.get("target_key_value"),
            rationale=raw.get("rationale", ""),
        )

    def instantiate_targets(self, decision: RoutingDecision, user_query: str) -> List[CandidateNode]:
        if decision.specific and decision.target_label and decision.target_key_name and decision.target_key_value:
            props = self.repo.get_node_context(
                decision.target_label,
                decision.target_key_name,
                decision.target_key_value,
            )
            return [
                CandidateNode(
                    label=decision.target_label,
                    key_name=decision.target_key_name,
                    key_value=decision.target_key_value,
                    display_name=self.repo.get_display_name(decision.target_label, props),
                    score=1.0,
                    summary="Specific target chosen by Fractal Manager",
                )
            ]

        if decision.applies_to_entire_scope:
            if decision.scope_level == "All":
                return self.repo.get_all_curriculum_nodes()
            return self.repo.get_nodes_by_level(decision.scope_level)

        matches = self.repo.search_candidates(user_query, limit_per_label=5000)
        if decision.scope_level == "All":
            return matches if matches else self.repo.get_all_curriculum_nodes()

        filtered = [m for m in matches if map_label_to_level(m.label) == decision.scope_level]
        if filtered:
            return filtered

        return self.repo.get_nodes_by_level(decision.scope_level)



# ORCHESTRATOR / GENERATOR / EVALUATOR / REPORTER
class Orchestrator:
    def __init__(self, llm: OpenAIService, archive: PromptArchive) -> None:
        self.llm = llm
        self.archive = archive

    def build_plan(
        self,
        bfu_dir: Path,
        user_query: str,
        node_label: str,
        node_props: Dict[str, Any],
        context_chain: Dict[str, Any],
    ) -> OrchestrationPlan:
        instructions = """
You are the Orchestrator inside a fractal self-organizing LLM-based multi-agent system.

For ONE curriculum node:
1. Define the generator role/profile prompt
2. Choose a model from the latest models offered by OpenAI (based on offical documentation) to use for the Generator based on the node's context and the user query in manner that prioritizes performance (produce reliable and accurate responses without producing references to hallucinated sources) while trying to manage costs.
3. Define strict evaluation criteria
4. Keep the result aligned to the selected node and its hierarchical context

The downstream Generator may propose graph mutations.
The downstream Evaluator will judge answer quality and node relevance.

Important:
- The Generator output format is fixed system-wide as canonical_graph_mutations_v1.
- Do not invent, vary, or redesign the Generator JSON schema.
- Do not embed alternate key names, examples, or wire-format instructions in the profile_prompt.
- Focus on role, domain constraints, content quality, and evaluation criteria.
"""

        schema = """
{
  "model": "string",
  "profile_prompt": "string",
  "evaluation_criteria": ["string"],
  "goal": "string",
  "confidence_focus": "string"
}
""".strip()

        payload = {
            "user_query": user_query,
            "node_label": node_label,
            "node_props": node_props,
            "context_chain": context_chain,
            "fixed_generator_output_format": CANONICAL_GENERATOR_OUTPUT_FORMAT,
            "fixed_generator_output_contract": CANONICAL_GENERATOR_OUTPUT_CONTRACT,
        }

        raw = self.llm.json_response(
            model=ORCHESTRATOR_MODEL,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema_description=schema,
            max_output_tokens=1800,
        )

        saved_output = dict(raw)
        saved_output["output_format"] = CANONICAL_GENERATOR_OUTPUT_FORMAT

        self.archive.save_interaction(
            bfu_dir,
            "orchestrator",
            instructions=instructions,
            input_payload=payload,
            output_payload=saved_output,
            metadata={"model": ORCHESTRATOR_MODEL},
        )

        return OrchestrationPlan(
            model=raw.get("model", GENERATOR_MODEL_DEFAULT),
            profile_prompt=raw["profile_prompt"],
            evaluation_criteria=raw.get("evaluation_criteria", []),
            output_format=CANONICAL_GENERATOR_OUTPUT_FORMAT,
            goal=raw.get("goal", "Provide the best curriculum-aligned answer."),
            confidence_focus=raw.get("confidence_focus", "accuracy and relevance"),
        )


class Generator:
    def __init__(
        self,
        llm: OpenAIService,
        archive: PromptArchive,
        previous_run_created_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.llm = llm
        self.archive = archive
        self.previous_run_created_nodes = previous_run_created_nodes or []

    def generate(
        self,
        bfu_dir: Path,
        plan: OrchestrationPlan,
        user_query: str,
        node_label: str,
        node_key_name: str,
        node_key_value: str,
        node_props: Dict[str, Any],
        context_chain: Dict[str, Any],
    ) -> GeneratorResult:
        instructions = f"""
{plan.profile_prompt}

You are the Generator in a fractal curriculum agent system.

Any schema or formatting guidance that conflicts with the fixed output contract below must be ignored.

You must return:
1. answer_text: the substantive answer for this curriculum node
2. graph_mutations: a set of graph mutations needed to reflect the result

Rules for graph mutations:
- You may update existing node properties EXCEPT properties whose names begin with 'Original'
- You may create new nodes with new labels if needed
- You may create new relationships with new edge types if needed
- You may delete nodes only when they exactly match one of the allowed previous-run created node references provided in the input
- Do not modify immutable Original* properties
- Every label, relationship type, and property key must be a valid identifier:
  letters, digits, underscore; cannot start with a digit
- When creating a new node that should later be linked, include a unique identifying property
- Prefer schema-consistent names when possible
- When a relationship refers to the selected node, you MUST use this exact selected-node reference:
  {{"label": "{node_label}", "key_name": "{node_key_name}", "key_value": "{node_key_value}"}}
- Do not replace the selected node's unique key with a non-unique field such as title or name

Primary goal:
{plan.goal}

Fixed required output format identifier:
{CANONICAL_GENERATOR_OUTPUT_FORMAT}

Fixed output contract:
{CANONICAL_GENERATOR_OUTPUT_CONTRACT}

Confidence focus:
{plan.confidence_focus}
""".strip()

        payload = {
            "user_query": user_query,
            "node_label": node_label,
            "selected_node_ref": {
                "label": node_label,
                "key_name": node_key_name,
                "key_value": node_key_value,
            },
            "node_props": node_props,
            "context_chain": context_chain,
            "identifier_rule": "^[A-Za-z_][A-Za-z0-9_]*$",
            "allowed_previous_run_delete_nodes": self.previous_run_created_nodes,
            "fixed_output_format": CANONICAL_GENERATOR_OUTPUT_FORMAT,
            "fixed_output_contract": CANONICAL_GENERATOR_OUTPUT_CONTRACT,
        }

        raw = self.llm.json_response(
            model=plan.model or GENERATOR_MODEL_DEFAULT,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema_description=CANONICAL_GENERATOR_JSON_SCHEMA,
            max_output_tokens=3200,
        )

        self.archive.save_interaction(
            bfu_dir,
            "generator",
            instructions=instructions,
            input_payload=payload,
            output_payload=raw,
            metadata={"model": plan.model or GENERATOR_MODEL_DEFAULT},
        )

        mutations = normalize_graph_mutations(extract_graph_mutations_payload(raw))
        return GeneratorResult(
            answer_text=raw.get("answer_text", ""),
            graph_mutations=GraphMutationPlan(
                updates=mutations.get("updates", []),
                create_nodes=mutations.get("create_nodes", []),
                create_relationships=mutations.get("create_relationships", []),
                delete_nodes=mutations.get("delete_nodes", []),
            ),
        )


class Evaluator:
    def __init__(self, llm: OpenAIService, archive: PromptArchive) -> None:
        self.llm = llm
        self.archive = archive

    def evaluate(
        self,
        bfu_dir: Path,
        user_query: str,
        node_label: str,
        node_props: Dict[str, Any],
        plan: OrchestrationPlan,
        generator_output: Dict[str, Any],
    ) -> EvaluationResult:
        instructions = """
You are the Evaluator in a fractal self-organizing curriculum agent system.

Evaluate the Generator's full output against:
- the user query
- the selected curriculum node
- the Orchestrator's evaluation criteria

The Generator output includes:
1. answer_text
2. graph_mutations

You must evaluate BOTH:
- whether the answer_text is useful, relevant, and aligned
- whether the graph_mutations correctly and completely reflect the intended changes

Evaluation expectations for graph_mutations:
- created nodes should be relevant to the selected node and the user query
- created relationships should correctly connect the intended nodes
- updates should target appropriate editable properties
- mutations should be consistent with the described answer_text
- proposed labels, relationship types, and property keys should appear valid and sensible
- mutations should not attempt to modify immutable Original* properties
- assess completeness, correctness, and alignment, not just prose quality

Scoring:
- score must be between 0 and 1
- passed=true only if the overall generator output is useful, aligned to the node, and meets the main criteria

Be strict and concrete.
""".strip()

        schema = """
{
  "score": 0.0,
  "passed": true,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "improvement_actions": ["string"],
  "rationale": "string"
}
""".strip()

        payload = {
            "user_query": user_query,
            "node_label": node_label,
            "node_props": node_props,
            "evaluation_criteria": plan.evaluation_criteria,
            "generator_output": generator_output,
        }

        raw = self.llm.json_response(
            model=EVALUATOR_MODEL,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema_description=schema,
            max_output_tokens=1400,
        )

        self.archive.save_interaction(
            bfu_dir,
            "evaluator",
            instructions=instructions,
            input_payload=payload,
            output_payload=raw,
            metadata={"model": EVALUATOR_MODEL},
        )

        score = float(raw.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=bool(raw.get("passed", False)),
            strengths=raw.get("strengths", []),
            weaknesses=raw.get("weaknesses", []),
            improvement_actions=raw.get("improvement_actions", []),
            rationale=raw.get("rationale", ""),
        )


class Reporter:
    def __init__(
        self,
        repo: Neo4jRepository,
        archive: PromptArchive,
        previous_run_created_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.repo = repo
        self.archive = archive
        self.previous_run_created_nodes = previous_run_created_nodes or []

    def _allowed_delete_refs(self) -> set[Tuple[str, str, str]]:
        refs = set()
        for node in self.previous_run_created_nodes:
            try:
                label = validate_identifier(node["label"], "label")
                key_name = validate_identifier(node["key_name"], "property key")
                key_value = str(node["key_value"])
                refs.add((label, key_name, key_value))
            except Exception:
                continue
        return refs

    @staticmethod
    def _validate_node_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": validate_identifier(ref["label"], "label"),
            "key_name": validate_identifier(ref["key_name"], "property key"),
            "key_value": ref["key_value"],
        }

    @staticmethod
    def _remember_temp_node_ref(
        temp_node_refs: Dict[str, Dict[str, Any]],
        raw_node: Dict[str, Any],
        created_ref: Dict[str, Any],
    ) -> None:
        for key in ("temp_id", "ref"):
            value = raw_node.get(key)
            if value not in (None, ""):
                temp_node_refs[str(value)] = dict(created_ref)

    def _resolve_node_ref(
        self,
        ref: Dict[str, Any],
        temp_node_refs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        temp_ref = ref.get("temp_id") or ref.get("ref")
        if temp_ref not in (None, ""):
            resolved = temp_node_refs.get(str(temp_ref))
            if resolved is None:
                raise ValueError(f"Unknown temp node reference: {temp_ref}")
            return dict(resolved)
        return self._validate_node_ref(ref)

    @staticmethod
    def _bind_selected_node_source_ref(
        ref: Dict[str, Any],
        execution: BFUExecution,
    ) -> Dict[str, Any]:
        if not isinstance(ref, dict):
            return ref
        if ref.get("temp_id") not in (None, "") or ref.get("ref") not in (None, ""):
            return ref

        label = ref.get("label")
        key_name = ref.get("key_name")
        key_value = ref.get("key_value")
        if label != execution.node_label:
            return ref
        if key_name == execution.node_key_name and str(key_value) == str(execution.node_key_value):
            return ref

        logger.warning(
            "Correcting ambiguous source reference for execution %s from %s(%s=%s) to %s(%s=%s)",
            execution.execution_id,
            label,
            key_name,
            key_value,
            execution.node_label,
            execution.node_key_name,
            execution.node_key_value,
        )
        return {
            "label": execution.node_label,
            "key_name": execution.node_key_name,
            "key_value": execution.node_key_value,
        }

    def report(self, bfu_dir: Path, execution: BFUExecution) -> Dict[str, Any]:
        mutation_results = self.apply_graph_mutations(execution)

        entry = {
            "record_type": "BFUExecution",
            "execution_id": execution.execution_id,
            "timestamp_utc": execution.timestamp_utc,
            "level": execution.level,
            "node_label": execution.node_label,
            "node_key_name": execution.node_key_name,
            "node_key_value": execution.node_key_value,
            "node_name": execution.node_name,
            "user_query": execution.user_query,
            "plan": asdict(execution.plan),
            "generator_output": execution.generator_output,
            "graph_mutations": execution.graph_mutations,
            "mutation_results": mutation_results,
            "evaluation": asdict(execution.evaluation),
            "propagated": execution.propagated,
            "propagation_source_execution_id": execution.propagation_source_execution_id,
        }

        self.repo.append_agent_log(
            execution.node_label,
            execution.node_key_name,
            execution.node_key_value,
            entry,
        )

        self.archive.save_interaction(
            bfu_dir,
            "reporter",
            instructions="Apply graph mutations, then append provenance into AgentLog.",
            input_payload={"execution": asdict(execution)},
            output_payload={
                "mutation_results": mutation_results,
                "agent_log_appended_to": {
                    "label": execution.node_label,
                    "key_name": execution.node_key_name,
                    "key_value": execution.node_key_value,
                },
            },
            metadata={"component": "reporter"},
        )

        return mutation_results

    def apply_graph_mutations(self, execution: BFUExecution) -> Dict[str, Any]:
        applied_updates = []
        applied_creates = []
        applied_relationships = []
        applied_deletes = []
        rejected = []
        temp_node_refs: Dict[str, Dict[str, Any]] = {}

        mutations = normalize_graph_mutations(execution.graph_mutations)
        allowed_delete_refs = self._allowed_delete_refs()

        for node in mutations.get("delete_nodes", []):
            try:
                ref = self._validate_node_ref(node)
                delete_key = (ref["label"], ref["key_name"], str(ref["key_value"]))
                if delete_key not in allowed_delete_refs:
                    raise ValueError("Deletion is only allowed for nodes created during the previous run.")

                self.repo.delete_node(
                    label=ref["label"],
                    key_name=ref["key_name"],
                    key_value=ref["key_value"],
                )

                applied_deletes.append(ref)
            except Exception as exc:
                rejected.append({
                    "type": "delete_node",
                    "reason": str(exc),
                    "payload": node,
                })

        for upd in mutations.get("updates", []):
            try:
                target = self._validate_node_ref(upd["target"])
                props = upd.get("properties", {}) or {}

                filtered, blocked = self.repo.update_node_properties(
                    label=target["label"],
                    key_name=target["key_name"],
                    key_value=target["key_value"],
                    properties=props,
                )

                if blocked:
                    rejected.append({
                        "type": "update",
                        "reason": "Attempted to modify immutable Original* properties",
                        "target": target,
                        "blocked_keys": blocked,
                    })

                if filtered:
                    applied_updates.append({
                        "target": target,
                        "properties": filtered,
                    })

            except Exception as exc:
                rejected.append({
                    "type": "update",
                    "reason": str(exc),
                    "payload": upd,
                })

        for node in mutations.get("create_nodes", []):
            try:
                label = validate_identifier(node["label"], "label")
                properties = node.get("properties", {}) or {}

                filtered_props, blocked = deep_filter_original_props(properties)
                identity_key, identity_value = select_identity_property(filtered_props)
                self.repo.create_node(label=label, properties=filtered_props)

                if blocked:
                    rejected.append({
                        "type": "create_node",
                        "reason": "Original* properties are not allowed on created nodes",
                        "blocked_keys": blocked,
                        "payload": node,
                    })

                created_ref = {
                    "label": label,
                    "properties": filtered_props,
                    "key_name": identity_key,
                    "key_value": identity_value,
                }
                self._remember_temp_node_ref(temp_node_refs, node, created_ref)
                applied_creates.append(created_ref)

            except Exception as exc:
                rejected.append({
                    "type": "create_node",
                    "reason": str(exc),
                    "payload": node,
                })

        for rel in mutations.get("create_relationships", []):
            try:
                source_ref = self._bind_selected_node_source_ref(rel["source"], execution)
                source = self._resolve_node_ref(source_ref, temp_node_refs)
                target = self._resolve_node_ref(rel["target"], temp_node_refs)
                rel_type = validate_identifier(rel["rel_type"], "relationship type")

                self.repo.create_relationship(
                    source_label=source["label"],
                    source_key_name=source["key_name"],
                    source_key_value=source["key_value"],
                    rel_type=rel_type,
                    target_label=target["label"],
                    target_key_name=target["key_name"],
                    target_key_value=target["key_value"],
                )

                applied_relationships.append({
                    "source": source,
                    "rel_type": rel_type,
                    "target": target,
                })

            except Exception as exc:
                rejected.append({
                    "type": "create_relationship",
                    "reason": str(exc),
                    "payload": rel,
                })

        return {
            "applied_updates": applied_updates,
            "applied_creates": applied_creates,
            "applied_relationships": applied_relationships,
            "applied_deletes": applied_deletes,
            "rejected": rejected,
        }



# FRACTAL UNIT
class FractalUnit:
    def __init__(
        self,
        llm: OpenAIService,
        repo: Neo4jRepository,
        archive: PromptArchive,
        previous_run_created_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.repo = repo
        self.archive = archive
        self.orchestrator = Orchestrator(llm, archive)
        self.generator = Generator(llm, archive, previous_run_created_nodes)
        self.evaluator = Evaluator(llm, archive)
        self.reporter = Reporter(repo, archive, previous_run_created_nodes)

    def run(
        self,
        *,
        user_query: str,
        node_label: str,
        key_name: str,
        key_value: str,
        propagated: bool = False,
        propagation_source_execution_id: Optional[str] = None,
        forced_plan: Optional[OrchestrationPlan] = None,
    ) -> BFUExecution:
        execution_id = str(uuid.uuid4())
        node_props = self.repo.get_node_context(node_label, key_name, key_value)
        context_chain = self.repo.get_parent_chain(node_label, key_name, key_value)
        node_name = self.repo.get_display_name(node_label, node_props)
        bfu_dir = self.archive.bfu_dir(node_name, execution_id)

        if forced_plan is None:
            plan = self.orchestrator.build_plan(
                bfu_dir=bfu_dir,
                user_query=user_query,
                node_label=node_label,
                node_props=node_props,
                context_chain=context_chain,
            )
        else:
            plan = forced_plan
            self.archive.save_interaction(
                bfu_dir,
                "orchestrator_reused_plan",
                instructions="Using propagated/redesigned orchestration plan.",
                input_payload={
                    "user_query": user_query,
                    "node_label": node_label,
                    "node_props": node_props,
                    "context_chain": context_chain,
                },
                output_payload=asdict(plan),
                metadata={"source": "fractal_level_manager"},
            )

        generator_result = self.generator.generate(
            bfu_dir=bfu_dir,
            plan=plan,
            user_query=user_query,
            node_label=node_label,
            node_key_name=key_name,
            node_key_value=key_value,
            node_props=node_props,
            context_chain=context_chain,
        )

        evaluation = self.evaluator.evaluate(
            bfu_dir=bfu_dir,
            user_query=user_query,
            node_label=node_label,
            node_props=node_props,
            plan=plan,
            generator_output={
                "answer_text": generator_result.answer_text,
                "graph_mutations": asdict(generator_result.graph_mutations),
            },
        )

        execution = BFUExecution(
            execution_id=execution_id,
            level=map_label_to_level(node_label),
            node_label=node_label,
            node_key_name=key_name,
            node_key_value=key_value,
            node_name=node_name,
            user_query=user_query,
            plan=plan,
            generator_output=generator_result.answer_text,
            graph_mutations=asdict(generator_result.graph_mutations),
            evaluation=evaluation,
            propagated=propagated,
            propagation_source_execution_id=propagation_source_execution_id,
        )

        execution.mutation_results = self.reporter.report(bfu_dir, execution)
        return execution



# FRACTAL LEVEL MANAGER
class FractalLevelManager:
    def __init__(
        self,
        llm: OpenAIService,
        repo: Neo4jRepository,
        archive: PromptArchive,
        previous_run_created_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.llm = llm
        self.repo = repo
        self.archive = archive
        self.previous_run_created_nodes = previous_run_created_nodes or []

    @staticmethod
    def _current_run_node_refs(executions: List[BFUExecution]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        seen: set[Tuple[str, str, str, str]] = set()

        for execution in executions:
            identity = (
                execution.level,
                execution.node_label,
                execution.node_key_name,
                str(execution.node_key_value),
            )
            if identity in seen:
                continue
            seen.add(identity)
            grouped.setdefault(execution.level, []).append({
                "node_label": execution.node_label,
                "node_key_name": execution.node_key_name,
                "node_key_value": str(execution.node_key_value),
                "node_name": execution.node_name,
                "current_execution_id": execution.execution_id,
            })

        return grouped

    @staticmethod
    def _build_propagation_record(raw: Dict[str, Any]) -> Optional[PropagationRecord]:
        execution_id = str(raw.get("execution_id") or "").strip()
        node_label = str(raw.get("node_label") or "").strip()
        node_key_name = str(raw.get("node_key_name") or "").strip()
        node_key_value = str(raw.get("node_key_value") or "").strip()
        if not execution_id or not node_label or not node_key_name or not node_key_value:
            return None

        plan = raw.get("plan", {}) if isinstance(raw.get("plan"), dict) else {}
        evaluation = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}

        try:
            score = float(evaluation.get("score", 0.0))
        except Exception:
            score = 0.0

        return PropagationRecord(
            execution_id=execution_id,
            node_label=node_label,
            node_key_name=node_key_name,
            node_key_value=node_key_value,
            node_name=str(raw.get("node_name") or node_key_value),
            level=str(raw.get("level") or map_label_to_level(node_label)),
            user_query=str(raw.get("user_query") or ""),
            propagated=bool(raw.get("propagated", False)),
            propagation_source_execution_id=raw.get("propagation_source_execution_id"),
            timestamp_utc=str(raw.get("timestamp_utc") or ""),
            model=str(plan.get("model") or ""),
            profile_prompt=str(plan.get("profile_prompt") or ""),
            evaluation_criteria=coerce_string_list(plan.get("evaluation_criteria")),
            output_format=str(plan.get("output_format") or ""),
            goal=str(plan.get("goal") or ""),
            confidence_focus=str(plan.get("confidence_focus") or ""),
            score=max(0.0, min(1.0, score)),
            passed=bool(evaluation.get("passed", False)),
            strengths=coerce_string_list(evaluation.get("strengths")),
            weaknesses=coerce_string_list(evaluation.get("weaknesses")),
            improvement_actions=coerce_string_list(evaluation.get("improvement_actions")),
            rationale=str(evaluation.get("rationale") or ""),
        )

    def _collect_level_evidence(self, level: str) -> List[PropagationRecord]:
        records = self.repo.get_agentlog_bfu_records_for_level(level)
        evidence: List[PropagationRecord] = []
        seen_execution_ids: set[str] = set()

        for raw in records:
            record = self._build_propagation_record(raw)
            if record is None or record.execution_id in seen_execution_ids:
                continue
            if record.level != level:
                continue
            seen_execution_ids.add(record.execution_id)
            evidence.append(record)

        return evidence

    def _analyze_level_history(
        self,
        *,
        mdir: Path,
        user_query: str,
        level: str,
        evidence: List[PropagationRecord],
        current_run_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        instructions = """
You are a Fractal Level Manager performing same-level change propagation.

Analyze the full AgentLog-derived BFU execution history for one level.

Rules:
- Use the provided records as the complete propagation evidence set
- Do not assume any record is pre-classified as strong or weak
- Infer success patterns and failure or inconsistency patterns from scores, passed flags, strengths, weaknesses, improvement actions, propagated status, and redesign lineage
- Recommend at most one universal redesign for the level
- Only select rerun targets from current_run_nodes
- Avoid overfitting to one node; look for repeated patterns that broadly help this level
- Stay aligned to the current user_query
- The Generator output format is fixed system-wide as canonical_graph_mutations_v1 and must not be changed
- If history is sparse or inconsistent, prefer propagation_warranted=false

Return ONLY JSON.
""".strip()

        schema = """
{
  "propagation_warranted": true,
  "universal_redesign": {
    "recommended_model": "string",
    "recommended_profile_prompt": "string",
    "recommended_evaluation_criteria": ["string"],
    "recommended_goal": "string",
    "recommended_confidence_focus": "string"
  },
  "nodes_to_rerun": [
    {
      "node_label": "string",
      "node_key_name": "string",
      "node_key_value": "string",
      "reason": "string"
    }
  ],
  "analysis_summary": {
    "observed_success_patterns": ["string"],
    "observed_failure_patterns": ["string"],
    "rationale": "string"
  }
}
""".strip()

        payload = {
            "user_query": user_query,
            "level": level,
            "agentlog_bfu_records": [asdict(record) for record in evidence],
            "current_run_nodes": current_run_nodes,
            "fixed_generator_output_format": CANONICAL_GENERATOR_OUTPUT_FORMAT,
        }

        analysis = self.llm.json_response(
            model=CHANGE_MANAGER_MODEL,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema_description=schema,
            max_output_tokens=2200,
        )

        self.archive.save_interaction(
            mdir,
            f"level_analysis_{level}",
            instructions=instructions,
            input_payload=payload,
            output_payload=analysis,
            metadata={"model": CHANGE_MANAGER_MODEL, "component": "fractal_level_manager"},
        )

        return analysis

    def plan_level_propagations(
        self,
        *,
        user_query: str,
        executions: List[BFUExecution],
    ) -> List[Dict[str, Any]]:
        manager_id = uuid.uuid4().hex[:10]
        mdir = self.archive.manager_dir("fractal_level_manager", manager_id)
        grouped_nodes = self._current_run_node_refs(executions)
        plans: List[Dict[str, Any]] = []

        for level, current_run_nodes in grouped_nodes.items():
            evidence = self._collect_level_evidence(level)
            analysis = self._analyze_level_history(
                mdir=mdir,
                user_query=user_query,
                level=level,
                evidence=evidence,
                current_run_nodes=current_run_nodes,
            )

            selected_nodes: List[Dict[str, Any]] = []
            seen_nodes: set[Tuple[str, str, str]] = set()
            current_node_lookup = {
                (
                    str(node["node_label"]),
                    str(node["node_key_name"]),
                    str(node["node_key_value"]),
                ): node
                for node in current_run_nodes
            }

            for raw_node in analysis.get("nodes_to_rerun", []) or []:
                if not isinstance(raw_node, dict):
                    continue
                identity = (
                    str(raw_node.get("node_label") or ""),
                    str(raw_node.get("node_key_name") or ""),
                    str(raw_node.get("node_key_value") or ""),
                )
                if identity in seen_nodes or identity not in current_node_lookup:
                    continue
                seen_nodes.add(identity)
                selected_nodes.append({
                    **current_node_lookup[identity],
                    "reason": str(raw_node.get("reason") or ""),
                })

            plans.append({
                "level": level,
                "manager_dir": str(mdir),
                "agentlog_evidence": evidence,
                "agentlog_record_count": len(evidence),
                "analysis": analysis,
                "current_run_nodes": current_run_nodes,
                "selected_nodes": selected_nodes,
            })

        return plans

    def propagation_needed(
        self,
        *,
        user_query: str,
        executions: List[BFUExecution],
        threshold: float,
    ) -> bool:
        _ = threshold
        plans = self.plan_level_propagations(user_query=user_query, executions=executions)
        return any(
            bool(plan["analysis"].get("propagation_warranted")) and plan["selected_nodes"]
            for plan in plans
        )

    def adapt_weak_executions(
        self,
        *,
        user_query: str,
        executions: List[BFUExecution],
        propagation_plans: List[Dict[str, Any]],
    ) -> List[BFUExecution]:
        execution_lookup = {
            (e.node_label, e.node_key_name, str(e.node_key_value)): e
            for e in executions
        }
        adapted_results: List[BFUExecution] = []

        for plan in propagation_plans:
            analysis = plan.get("analysis", {}) or {}
            if not analysis.get("propagation_warranted"):
                continue
            universal_redesign = analysis.get("universal_redesign")
            if not isinstance(universal_redesign, dict) or not plan.get("selected_nodes"):
                continue

            mdir = Path(str(plan["manager_dir"]))
            for selected_node in plan["selected_nodes"]:
                identity = (
                    str(selected_node["node_label"]),
                    str(selected_node["node_key_name"]),
                    str(selected_node["node_key_value"]),
                )
                current_execution = execution_lookup.get(identity)
                if current_execution is None:
                    logger.warning("Skipping rerun target missing from current invocation: %s", selected_node)
                    continue

                adapted = self._redesign_and_rerun(
                    mdir=mdir,
                    user_query=user_query,
                    level=str(plan["level"]),
                    current_execution=current_execution,
                    redesign=universal_redesign,
                    analysis_summary=analysis.get("analysis_summary", {}) or {},
                    agentlog_evidence=plan.get("agentlog_evidence", []),
                    current_run_nodes=plan.get("current_run_nodes", []),
                    selected_nodes=plan.get("selected_nodes", []),
                    rerun_reason=str(selected_node.get("reason") or ""),
                )
                adapted_results.append(adapted)

        return adapted_results

    def _redesign_and_rerun(
        self,
        *,
        mdir: Path,
        user_query: str,
        level: str,
        current_execution: BFUExecution,
        redesign: Dict[str, Any],
        analysis_summary: Dict[str, Any],
        agentlog_evidence: List[PropagationRecord],
        current_run_nodes: List[Dict[str, Any]],
        selected_nodes: List[Dict[str, Any]],
        rerun_reason: str,
    ) -> BFUExecution:
        redesign_dir = mdir / f"redesign_{current_execution.execution_id}"
        ensure_dir(redesign_dir)

        self.archive.save_interaction(
            redesign_dir,
            "redesign_application",
            instructions="Apply the universal same-level redesign selected from AgentLog analysis to one current-run node.",
            input_payload={
                "user_query": user_query,
                "level": level,
                "target_execution_id": current_execution.execution_id,
                "target_node": {
                    "node_label": current_execution.node_label,
                    "node_key_name": current_execution.node_key_name,
                    "node_key_value": current_execution.node_key_value,
                    "node_name": current_execution.node_name,
                },
                "rerun_reason": rerun_reason,
                "universal_redesign": redesign,
                "analysis_summary": analysis_summary,
            },
            output_payload=redesign,
            metadata={"component": "fractal_level_manager"},
        )

        new_plan = OrchestrationPlan(
            model=redesign.get("recommended_model", current_execution.plan.model),
            profile_prompt=redesign.get("recommended_profile_prompt", current_execution.plan.profile_prompt),
            evaluation_criteria=redesign.get(
                "recommended_evaluation_criteria",
                current_execution.plan.evaluation_criteria,
            ),
            output_format=CANONICAL_GENERATOR_OUTPUT_FORMAT,
            goal=redesign.get("recommended_goal", current_execution.plan.goal),
            confidence_focus=redesign.get(
                "recommended_confidence_focus",
                current_execution.plan.confidence_focus,
            ),
        )

        bfu = FractalUnit(
            self.llm,
            self.repo,
            self.archive,
            previous_run_created_nodes=self.previous_run_created_nodes,
        )
        adapted_execution = bfu.run(
            user_query=user_query,
            node_label=current_execution.node_label,
            key_name=current_execution.node_key_name,
            key_value=current_execution.node_key_value,
            propagated=True,
            propagation_source_execution_id=current_execution.execution_id,
            forced_plan=new_plan,
        )

        self.repo.append_agent_log(
            current_execution.node_label,
            current_execution.node_key_name,
            current_execution.node_key_value,
            {
                "record_type": "ChangePropagation",
                "timestamp_utc": utc_now_iso(),
                "level": level,
                "origin_execution_id": current_execution.execution_id,
                "new_execution_id": adapted_execution.execution_id,
                "node_name": current_execution.node_name,
                "user_query": user_query,
                "redesign": asdict(new_plan),
                "analysis_summary": analysis_summary,
                "agentlog_record_count": len(agentlog_evidence),
                "analyzed_execution_ids": [record.execution_id for record in agentlog_evidence],
                "current_run_nodes_considered": current_run_nodes,
                "nodes_selected_for_rerun": selected_nodes,
                "selected_node_reason": rerun_reason,
                "old_score": current_execution.evaluation.score,
                "new_score": adapted_execution.evaluation.score,
                "rationale": analysis_summary.get("rationale", ""),
            },
        )

        return adapted_execution



# SYSTEM FACADE
class FSALMaS:
    def __init__(self) -> None:
        self.llm = OpenAIService()
        self.repo = Neo4jRepository()
        self.archive = PromptArchive()
        self.run_state = RunStateStore()
        self.previous_run_created_nodes = self.run_state.load_previous_created_nodes()
        self.fractal_manager = FractalManager(self.llm, self.repo, self.archive)
        self.level_manager = FractalLevelManager(
            self.llm,
            self.repo,
            self.archive,
            previous_run_created_nodes=self.previous_run_created_nodes,
        )

    def close(self) -> None:
        self.repo.close()

    def answer_query(
        self,
        user_query: str,
        propagation_threshold: float = PROPAGATION_THRESHOLD,
    ) -> Dict[str, Any]:
        decision = self.fractal_manager.route_query(user_query)
        targets = self.fractal_manager.instantiate_targets(decision, user_query)

        if not targets:
            self.run_state.save_current_run(
                run_id=self.archive.run_id,
                run_dir=str(self.archive.run_dir.resolve()),
                created_nodes=[],
            )
            return {
                "routing": asdict(decision),
                "targets": [],
                "executions": [],
                "propagated_executions": [],
                "final_answer": "No relevant curriculum elements were found.",
                "prompt_log_run_dir": str(self.archive.run_dir.resolve()),
            }

        bfu = FractalUnit(
            self.llm,
            self.repo,
            self.archive,
            previous_run_created_nodes=self.previous_run_created_nodes,
        )
        initial_executions: List[BFUExecution] = []

        for target in targets:
            logger.info(
                "Running BFU for %s(%s=%s)",
                target.label,
                target.key_name,
                target.key_value,
            )
            execution = bfu.run(
                user_query=user_query,
                node_label=target.label,
                key_name=target.key_name,
                key_value=target.key_value,
            )
            initial_executions.append(execution)

        propagated_executions: List[BFUExecution] = []
        propagation_plans = self.level_manager.plan_level_propagations(
            user_query=user_query,
            executions=initial_executions,
        )
        propagation_needed = any(
            bool(plan["analysis"].get("propagation_warranted")) and plan["selected_nodes"]
            for plan in propagation_plans
        )
        if propagation_needed:
            if self._confirm_change_propagation():
                propagated_executions = self.level_manager.adapt_weak_executions(
                    user_query=user_query,
                    executions=initial_executions,
                    propagation_plans=propagation_plans,
                )
            else:
                logger.info("Change propagation stopped by user input.")

        best_by_node: Dict[Tuple[str, str, str], BFUExecution] = {}
        for execution in initial_executions + propagated_executions:
            node_key = (execution.node_label, execution.node_key_name, execution.node_key_value)
            current = best_by_node.get(node_key)
            if current is None or execution.evaluation.score > current.evaluation.score:
                best_by_node[node_key] = execution

        final_executions = list(best_by_node.values())

        synthesis = self._synthesize_outputs(
            user_query=user_query,
            decision=decision,
            executions=final_executions,
            initial_executions=initial_executions,
            propagated_executions=propagated_executions,
        )

        current_run_created_nodes = self._collect_created_nodes(initial_executions + propagated_executions)
        self.run_state.save_current_run(
            run_id=self.archive.run_id,
            run_dir=str(self.archive.run_dir.resolve()),
            created_nodes=current_run_created_nodes,
        )

        return {
            "routing": asdict(decision),
            "targets": [asdict(t) for t in targets],
            "executions": [asdict(x) for x in final_executions],
            "initial_executions": [asdict(x) for x in initial_executions],
            "propagated_executions": [asdict(x) for x in propagated_executions],
            "final_answer": synthesis,
            "prompt_log_run_dir": str(self.archive.run_dir.resolve()),
            "previous_run_created_nodes_count": len(self.previous_run_created_nodes),
            "current_run_created_nodes_count": len(current_run_created_nodes),
        }

    @staticmethod
    def _confirm_change_propagation() -> bool:
        while True:
            choice = input("\nChange propagation is ready to run. Continue? [y/n]: ").strip().lower()
            if choice in {"y", "yes"}:
                return True
            if choice in {"n", "no"}:
                return False
            print("Please enter 'y' to continue or 'n' to stop.")

    @staticmethod
    def _collect_created_nodes(executions: List[BFUExecution]) -> List[Dict[str, Any]]:
        created_nodes: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        for execution in executions:
            mutation_results = execution.mutation_results or {}
            for created in mutation_results.get("applied_creates", []):
                try:
                    label = validate_identifier(created["label"], "label")
                    key_name = validate_identifier(created["key_name"], "property key")
                    key_value = created["key_value"]
                    properties = created.get("properties", {}) or {}
                    identity = (label, key_name, str(key_value))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    created_nodes.append({
                        "label": label,
                        "key_name": key_name,
                        "key_value": key_value,
                        "properties": properties,
                    })
                except Exception:
                    continue

        return created_nodes

    def _synthesize_outputs(
        self,
        *,
        user_query: str,
        decision: RoutingDecision,
        executions: List[BFUExecution],
        initial_executions: List[BFUExecution],
        propagated_executions: List[BFUExecution],
    ) -> str:
        sdir = self.archive.manager_dir("final_synthesis")

        instructions = """
You are the final synthesis layer of a fractal curriculum multi-agent system.

Combine the BFU outputs into one coherent answer.

Requirements:
- preserve curriculum relevance
- prefer the strongest execution for each node
- mention important propagation improvements if useful
- note major graph changes if they matter
- provide a clean final response for the user
"""

        payload = {
            "user_query": user_query,
            "routing_decision": asdict(decision),
            "selected_node_outputs": [
                {
                    "node_name": e.node_name,
                    "level": e.level,
                    "generator_output": e.generator_output,
                    "evaluation_score": e.evaluation.score,
                    "evaluation_passed": e.evaluation.passed,
                    "propagated": e.propagated,
                    "propagation_source_execution_id": e.propagation_source_execution_id,
                    "graph_mutations": e.graph_mutations,
                }
                for e in executions
            ],
            "initial_execution_count": len(initial_executions),
            "propagated_execution_count": len(propagated_executions),
        }

        output = self.llm.text_response(
            model=GENERATOR_MODEL_DEFAULT,
            instructions=instructions,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            max_output_tokens=2200,
        )

        self.archive.save_interaction(
            sdir,
            "synthesis",
            instructions=instructions,
            input_payload=payload,
            output_payload=output,
            metadata={"model": GENERATOR_MODEL_DEFAULT},
        )

        return output



# MAIN
def main() -> int:
    system = FSALMaS()

    try:
        result = system.answer_query(
            USER_QUERY,
            propagation_threshold=PROPAGATION_THRESHOLD,
        )

        print("\n=== USER QUERY ===")
        print(USER_QUERY)

        print("\n=== ROUTING DECISION ===")
        print(json.dumps(result["routing"], indent=2, ensure_ascii=False))

        print("\n=== TARGET COUNT ===")
        print(len(result.get("targets", [])))

        print("\n=== INITIAL EXECUTION COUNT ===")
        print(len(result.get("initial_executions", [])))

        print("\n=== PROPAGATED EXECUTION COUNT ===")
        print(len(result.get("propagated_executions", [])))

        print("\n=== PREVIOUS RUN CREATED NODE COUNT ===")
        print(result.get("previous_run_created_nodes_count", 0))

        print("\n=== CURRENT RUN CREATED NODE COUNT ===")
        print(result.get("current_run_created_nodes_count", 0))

        print("\n=== FINAL ANSWER ===")
        print(result["final_answer"])

        print("\n=== FINAL EXECUTION SUMMARY ===")
        summary = [
            {
                "node_name": e["node_name"],
                "level": e["level"],
                "score": e["evaluation"]["score"],
                "passed": e["evaluation"]["passed"],
                "propagated": e["propagated"],
            }
            for e in result["executions"]
        ]
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        print("\n=== PROMPT LOG DIRECTORY ===")
        print(result["prompt_log_run_dir"])

        return 0

    finally:
        system.close()


if __name__ == "__main__":
    raise SystemExit(main())
