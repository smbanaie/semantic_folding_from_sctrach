"""
Memgraph client for knowledge graph storage.

Uses Neo4j Bolt protocol to connect to Memgraph and perform
graph operations (create nodes, relationships, bulk inserts).
"""

import logging
from typing import Dict, List, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.models.data_models import Triple

logger = logging.getLogger(__name__)


class MemgraphClient:
    """
    Client for Memgraph graph database operations.

    Uses Neo4j driver (Memgraph is compatible with Neo4j Bolt protocol).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
    ):
        """
        Initialize Memgraph client.

        Args:
            uri: Memgraph connection URI (default: bolt://localhost:7687)
            user: Username (usually empty for Memgraph)
            password: Password (usually empty for Memgraph)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[Driver] = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def connect(self):
        """Establish connection to Memgraph."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password) if self.user else None,
            )
            # Verify connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Memgraph at {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            raise RuntimeError(
                f"Cannot connect to Memgraph at {self.uri}. "
                "Make sure Memgraph is running (docker run -p 7687:7687 memgraph/memgraph)"
            ) from e

    def close(self):
        """Close connection to Memgraph."""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Memgraph")

    def create_node(
        self,
        entity: str,
        entity_type: str = "Entity",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Create a node in the graph.

        Args:
            entity: Entity name/label
            entity_type: Type of entity (e.g., "Person", "Organization")
            metadata: Additional properties for the node
        """
        if not self.driver:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")

        metadata = metadata or {}
        properties = {
            "name": entity,
            "type": entity_type,
            **metadata,
        }

        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $type
        SET n += $properties
        """
        params = {
            "name": entity,
            "type": entity_type,
            "properties": {k: v for k, v in properties.items() if k not in ["name", "type"]},
        }

        with self.driver.session() as session:
            session.run(query, params)
            logger.debug(f"Created/updated node: {entity} ({entity_type})")

    def create_relationship(
        self,
        subject: str,
        predicate: str,
        object: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Create a relationship between two nodes.

        Args:
            subject: Subject entity name
            predicate: Relationship type/predicate
            object: Object entity name
            metadata: Additional properties for the relationship
        """
        if not self.driver:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")

        # Ensure nodes exist
        self.create_node(subject)
        self.create_node(object)

        # Create relationship
        metadata = metadata or {}
        properties = {
            "predicate": predicate,
            **metadata,
        }

        # Sanitize predicate for relationship type (Cypher doesn't allow spaces)
        rel_type = predicate.replace(" ", "_").upper()

        query = f"""
        MATCH (s:Entity {{name: $subject}})
        MATCH (o:Entity {{name: $object}})
        MERGE (s)-[r:{rel_type}]->(o)
        SET r += $properties
        """
        params = {
            "subject": subject,
            "object": object,
            "properties": properties,
        }

        with self.driver.session() as session:
            session.run(query, params)
            logger.debug(f"Created relationship: {subject} -[{predicate}]-> {object}")

    def bulk_insert_triples(self, triples: List[Triple]) -> None:
        """
        Bulk insert knowledge triples into the graph.

        Args:
            triples: List of Triple objects to insert
        """
        if not self.driver:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")

        if not triples:
            logger.warning("No triples to insert")
            return

        logger.info(f"Inserting {len(triples)} triples into Memgraph...")

        with self.driver.session() as session:
            # Use a transaction for bulk insert
            with session.begin_transaction() as tx:
                for triple in triples:
                    try:
                        # Create nodes
                        self._create_node_in_tx(tx, triple.subject, triple.metadata.get("subject_type", "Entity"))
                        self._create_node_in_tx(tx, triple.object, triple.metadata.get("object_type", "Entity"))

                        # Create relationship
                        rel_type = triple.predicate.replace(" ", "_").upper()
                        query = f"""
                        MATCH (s:Entity {{name: $subject}})
                        MATCH (o:Entity {{name: $object}})
                        MERGE (s)-[r:{rel_type}]->(o)
                        SET r.predicate = $predicate,
                            r.confidence = $confidence,
                            r.source_chunk_id = $source_chunk_id
                        """
                        params = {
                            "subject": triple.subject,
                            "object": triple.object,
                            "predicate": triple.predicate,
                            "confidence": triple.confidence,
                            "source_chunk_id": triple.source_chunk_id,
                        }
                        tx.run(query, params)

                    except Exception as e:
                        logger.error(f"Error inserting triple {triple}: {e}")
                        # Continue with next triple
                        continue

                tx.commit()

        logger.info(f"Successfully inserted {len(triples)} triples")

    def _create_node_in_tx(self, tx, entity: str, entity_type: str = "Entity"):
        """Helper to create node within a transaction."""
        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $type
        """
        tx.run(query, {"name": entity, "type": entity_type})

    def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node_count and edge_count
        """
        if not self.driver:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")

        with self.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]

            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            edge_count = rel_result.single()["count"]

        return {
            "node_count": node_count,
            "edge_count": edge_count,
        }

    def clear_graph(self) -> None:
        """
        Clear all nodes and relationships from the graph.

        Warning: This deletes all data!
        """
        if not self.driver:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")

        logger.warning("Clearing all data from Memgraph...")

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        logger.info("Graph cleared successfully")

