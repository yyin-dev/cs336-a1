import time
import pathlib
import cProfile

from src.bpe import train_bpe

# Run this with so that relative import works without path hack
# $ uv run -m profiling.train_bpe_speed


FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent).parent / "tests" / "fixtures"


def main():
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    # main()
    cProfile.run("main()", sort="tottime")
