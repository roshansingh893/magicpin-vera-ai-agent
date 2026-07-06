"""Dataset loader — unified data loading for evaluation.

Loads categories, merchants, triggers, and customers from the
challenge dataset directory.  Supports single-scenario, batch,
and full-dataset loading.

Never duplicates loading logic — all evaluation modules use this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default dataset path relative to project root
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


@dataclass
class EvaluationScenario:
    """A single (category, merchant, trigger, customer?) evaluation case.

    This is the unit of work for the batch runner.
    """
    test_id: str
    category: dict[str, Any]
    merchant: dict[str, Any]
    trigger: dict[str, Any]
    customer: Optional[dict[str, Any]] = None

    @property
    def merchant_id(self) -> str:
        return self.merchant.get("merchant_id", "unknown")

    @property
    def trigger_kind(self) -> str:
        return self.trigger.get("kind", "unknown")

    @property
    def category_slug(self) -> str:
        return self.category.get("slug", "unknown")


@dataclass
class Dataset:
    """Loaded dataset — all categories, merchants, triggers, customers."""
    categories: dict[str, dict[str, Any]] = field(default_factory=dict)
    merchants: list[dict[str, Any]] = field(default_factory=list)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    customers: list[dict[str, Any]] = field(default_factory=list)

    # Lookup indices
    _merchant_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    _customer_index: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build_indices(self) -> None:
        """Build fast lookup indices after loading."""
        self._merchant_index = {m["merchant_id"]: m for m in self.merchants}
        self._customer_index = {c["customer_id"]: c for c in self.customers}

    def get_merchant(self, merchant_id: str) -> dict[str, Any] | None:
        return self._merchant_index.get(merchant_id)

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return self._customer_index.get(customer_id)

    def get_category(self, slug: str) -> dict[str, Any] | None:
        return self.categories.get(slug)


def load_categories(dataset_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load all category JSON files.

    Returns:
        Dict mapping slug → category data.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    categories_dir = dataset_dir / "categories"
    categories: dict[str, dict[str, Any]] = {}

    if not categories_dir.exists():
        logger.warning("Categories directory not found: %s", categories_dir)
        return categories

    for f in sorted(categories_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        slug = data.get("slug", f.stem)
        categories[slug] = data
        logger.debug("Loaded category: %s", slug)

    logger.info("Loaded %d categories", len(categories))
    return categories


def load_merchants(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load merchant seed data.

    Returns:
        List of merchant dicts.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    seed_file = dataset_dir / "merchants_seed.json"

    if not seed_file.exists():
        logger.warning("Merchants seed file not found: %s", seed_file)
        return []

    data = json.loads(seed_file.read_text(encoding="utf-8"))
    merchants = data.get("merchants", [])
    logger.info("Loaded %d merchants", len(merchants))
    return merchants


def load_triggers(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load trigger seed data.

    Returns:
        List of trigger dicts.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    seed_file = dataset_dir / "triggers_seed.json"

    if not seed_file.exists():
        logger.warning("Triggers seed file not found: %s", seed_file)
        return []

    data = json.loads(seed_file.read_text(encoding="utf-8"))
    triggers = data.get("triggers", [])
    logger.info("Loaded %d triggers", len(triggers))
    return triggers


def load_customers(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load customer seed data.

    Returns:
        List of customer dicts.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    seed_file = dataset_dir / "customers_seed.json"

    if not seed_file.exists():
        logger.warning("Customers seed file not found: %s", seed_file)
        return []

    data = json.loads(seed_file.read_text(encoding="utf-8"))
    customers = data.get("customers", [])
    logger.info("Loaded %d customers", len(customers))
    return customers


def load_dataset(dataset_dir: Path | None = None) -> Dataset:
    """Load the complete dataset with all four context types.

    Returns:
        A Dataset object with built indices.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR

    ds = Dataset(
        categories=load_categories(dataset_dir),
        merchants=load_merchants(dataset_dir),
        triggers=load_triggers(dataset_dir),
        customers=load_customers(dataset_dir),
    )
    ds.build_indices()

    logger.info(
        "Full dataset loaded: %d categories, %d merchants, %d triggers, %d customers",
        len(ds.categories),
        len(ds.merchants),
        len(ds.triggers),
        len(ds.customers),
    )
    return ds


def build_scenarios(dataset: Dataset) -> list[EvaluationScenario]:
    """Build evaluation scenarios from the dataset.

    Each trigger maps to a merchant (and optionally a customer).
    The category is resolved from the merchant's category_slug.

    Returns:
        List of EvaluationScenario objects.
    """
    scenarios: list[EvaluationScenario] = []

    for i, trigger in enumerate(dataset.triggers, start=1):
        merchant_id = trigger.get("merchant_id", "")
        merchant = dataset.get_merchant(merchant_id)

        if not merchant:
            logger.warning("Skipping trigger %s — merchant %s not found", trigger["id"], merchant_id)
            continue

        category_slug = merchant.get("category_slug", "")
        category = dataset.get_category(category_slug)

        if not category:
            logger.warning("Skipping trigger %s — category %s not found", trigger["id"], category_slug)
            continue

        # Resolve customer if present
        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = dataset.get_customer(customer_id)

        test_id = f"T{i:02d}"
        scenarios.append(EvaluationScenario(
            test_id=test_id,
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        ))

    logger.info("Built %d evaluation scenarios from dataset", len(scenarios))
    return scenarios


def load_golden_scenario(filepath: Path) -> EvaluationScenario:
    """Load a single golden scenario from a JSON file.

    Args:
        filepath: Path to the golden scenario JSON.

    Returns:
        An EvaluationScenario.
    """
    data = json.loads(filepath.read_text(encoding="utf-8"))
    return EvaluationScenario(
        test_id=data.get("test_id", filepath.stem),
        category=data["category"],
        merchant=data["merchant"],
        trigger=data["trigger"],
        customer=data.get("customer"),
    )


def load_golden_scenarios(directory: Path | None = None) -> list[EvaluationScenario]:
    """Load all golden scenarios from a directory.

    Args:
        directory: Path to golden_scenarios/ dir.

    Returns:
        List of EvaluationScenarios.
    """
    if directory is None:
        directory = Path(__file__).resolve().parent.parent / "golden_scenarios"

    if not directory.exists():
        logger.warning("Golden scenarios directory not found: %s", directory)
        return []

    scenarios = []
    for f in sorted(directory.glob("*.json")):
        try:
            s = load_golden_scenario(f)
            scenarios.append(s)
            logger.debug("Loaded golden scenario: %s", s.test_id)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load golden scenario %s: %s", f.name, exc)

    logger.info("Loaded %d golden scenarios", len(scenarios))
    return scenarios
