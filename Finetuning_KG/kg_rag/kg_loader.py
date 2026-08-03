"""Load the clinical knowledge graph and build query indexes."""
import json
from collections import defaultdict


class ClinicalKG:
    """Indexed clinical knowledge graph."""

    def __init__(self, kg_path):
        """Load a knowledge graph and build indexes.

        Args:
            kg_path: Knowledge-graph JSON path.
        """
        with open(kg_path, encoding="utf-8") as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.triples = data["triples"]
        self._build_index()

    def _build_index(self):
        """Build adjacency and alias indexes."""
        # Adjacency list: spo[subject][predicate] = [triple, ...].
        self.spo = defaultdict(lambda: defaultdict(list))
        for t in self.triples:
            self.spo[t["s"]][t["p"]].append(t)

        # Alias index: alias -> canonical entity name.
        self.alias2entity = {}
        for name, attr in self.entities.items():
            self.alias2entity[name] = name
            for alias in attr.get("aliases", []):
                self.alias2entity[alias] = name

    def get_entity(self, name):
        """Return entity attributes.

        Args:
            name: Canonical entity name.

        Returns:
            Entity dictionary, or ``None`` when absent.
        """
        return self.entities.get(name)

    def get_objects(self, s, p):
        """Return objects for all ``(subject, predicate, ?)`` triples.

        Args:
            s: Subject entity.
            p: Relation predicate.

        Returns:
            Object entity names.
        """
        return [t["o"] for t in self.spo[s][p]]

    def get_triples(self, s, p):
        """Return complete triples for ``(subject, predicate, ?)``.

        Args:
            s: Subject entity.
            p: Relation predicate.

        Returns:
            Triple dictionaries, including optional metadata.
        """
        return self.spo[s][p]

    def entities_by_type(self, etype):
        """Return all entities of one type.

        Args:
            etype: Entity type, such as ``Measurement`` or ``Finding``.

        Returns:
            Canonical entity names.
        """
        return [name for name, attr in self.entities.items()
                if attr.get("type") == etype]

    def resolve_alias(self, alias):
        """Resolve an alias to its canonical entity name.

        Args:
            alias: Alias or canonical name.

        Returns:
            Canonical name, or ``None`` when absent.
        """
        return self.alias2entity.get(alias)
