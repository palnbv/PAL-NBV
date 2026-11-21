"""Local asset manifest; importing this module does not start Isaac Sim."""

from pathlib import Path

WAREHOUSE_DIR = "Environments/Simple_Warehouse"
WAREHOUSE = f"{WAREHOUSE_DIR}/warehouse.usd"
PALLET = f"{WAREHOUSE_DIR}/Props/SM_PaletteA_01.usd"
BLOCKERS = (
    "Props/KLT_Bin/small_KLT.usd",
    *(f"{WAREHOUSE_DIR}/Props/{name}.usd" for name in (
        "SM_CardBoxA_01_413", "SM_CardBoxA_01_611", "SM_CardBoxB_01_1190",
        "SM_CratePlasticNote_A_01", "SM_FireExtinguisher_02", "SM_CratePlastic_E_01",
    )),
)


def validate_assets(root: Path, model: Path) -> None:
    missing = [path for path in (model, *(root / name for name in (WAREHOUSE, PALLET, *BLOCKERS)))
               if not path.is_file()]
    if missing:
        raise ValueError("Missing input files (check --asset-root and --model):\n" + "\n".join(map(str, missing)))