import subprocess


def main():
    # subprocess.run(["python", "src/train_models.py",
    #                 "--data-dir", "./data/pet",
    #                 "--num-runs", "1",
    #                ])

    # subprocess.run(["python", "src/create_model_summary.py"])
    # subprocess.run(["python", "src/display_aggregated_results.py"])

    subprocess.run(["python", "src/xai_analysis.py"])
    subprocess.run(["python", "src/create_xai_summary.py"])


if __name__ == "__main__":
    main()
