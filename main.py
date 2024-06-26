import argparse
import pathlib

import pypdf
import numpy as np
import numpy.typing as npt

from src.algorithm.clustering import ClusteringModel
from src.algorithm.elbow import Elbow


def main() -> None:
    parser = argparse.ArgumentParser(description="Get the number of columns in a pdf file using clustering")

    parser.add_argument("resume_path", type=pathlib.Path, help="The absolute path to a pdf file")

    args = parser.parse_args()

    verify_file_path(args.resume_path)

    print(get_num_cols(text_to_data_points(extract_text_from_pdf(args.resume_path))))


def verify_file_path(file_path: pathlib.Path) -> None:
    file_path = file_path.resolve()

    if not file_path.exists():
        raise FileNotFoundError("File not found")


def extract_text_from_pdf(file_path: pathlib.Path) -> str:
    pdf_reader = pypdf.PdfReader(file_path)

    return "\n".join([page.extract_text(extraction_mode="layout") for page in pdf_reader.pages])


def text_to_data_points(text: str) -> npt.NDArray[np.int64]:
    character_position_list: list[tuple[int, int]] = []

    for y, line in enumerate(text.splitlines()):
        for x, char in enumerate(line):
            if not char.isspace():
                character_position_list.append((x, y))

    return np.array(character_position_list)


def get_num_cols(data: npt.NDArray[np.int64]) -> np.int64:
    model = ClusteringModel(data, 0, "lloyds")
    elbow = Elbow(model, 2, 6, data)
    return elbow.find_elbow()


if __name__ == "__main__":
    main()
