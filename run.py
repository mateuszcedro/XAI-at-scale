import subprocess

DATASET = "chest_x_pneumo"  # Options: "covid_qu_ex", "pet", "chest_x_pneumo"


def main():
    subprocess.run(["python", "src/train_models.py",
                    "--data-dir", f"./data/{DATASET}",
                   ])

    subprocess.run(["python", "src/create_model_summary.py"])
    subprocess.run(["python", "src/display_aggregated_results.py"])

    subprocess.run(["python", "src/xai_analysis.py"])
    subprocess.run(["python", "src/create_xai_summary.py"])


if __name__ == "__main__":
    main()
