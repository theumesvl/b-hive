import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc

"""
usage:
python clean_parquet.py \
  --filelist /eos/user/t/tvanlaer/higgscharm/outputs/hww/2022postEE_20260313_1530/filelists/base.txt \
  --columns dilepton_pt dilepton_mass mtl1 mtl2 cjet_cand_cvsl_pnet
"""
def build_valid_mask(table, columns):
    """
    Build a mask that is True only for rows where ALL specified columns
    are valid (not None and not NaN).
    """
    masks = []

    for colname in columns:
        if colname not in table.schema.names:
            print(f"Warning: column '{colname}' not found, skipping")
            continue

        col = table.column(colname)

        # valid (not None)
        is_valid = pc.is_valid(col)

        # not NaN (handle only float-like columns safely)
        not_nan = pc.invert(
            pc.is_nan(
                pc.if_else(is_valid, col, 0.0)
            )
        )

        masks.append(pc.and_(is_valid, not_nan))

    if not masks:
        raise ValueError("No valid columns found to build mask")

    # Combine all masks with AND → drop row if ANY column is bad
    combined = masks[0]
    for m in masks[1:]:
        combined = pc.and_(combined, m)

    return combined


def clean_file(path, columns):
    table = pq.read_table(path)

    mask = build_valid_mask(table, columns)
    cleaned = table.filter(mask)

    removed = table.num_rows - cleaned.num_rows

    if removed:
        pq.write_table(cleaned, path)
        print(f"{path}: removed {removed} rows ({table.num_rows} -> {cleaned.num_rows})")
    else:
        print(f"{path}: no NaN/null rows found, skipping")


def read_filelist(txt_path):
    with open(txt_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows with NaN/None in selected columns from parquet files"
    )
    parser.add_argument(
        "--filelist",
        required=True,
        help="Path to txt file containing parquet paths"
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Columns to check for NaN/None"
    )

    args = parser.parse_args()

    files = read_filelist(args.filelist)

    for f in files:
        clean_file(f, args.columns)


if __name__ == "__main__":
    main()
