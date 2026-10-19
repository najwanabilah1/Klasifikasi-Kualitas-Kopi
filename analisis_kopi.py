from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


URL_DATA = (
    "https://raw.githubusercontent.com/"
    "rfordatascience/tidytuesday/master/"
    "data/2020/2020-07-07/coffee_ratings.csv"
)
BASE_DIR = Path(__file__).resolve().parent
PATH_DATA = BASE_DIR / "coffee_ratings.csv"
DIR_OUTPUT = BASE_DIR / "output"
AMBANG_KUALITAS = 80
RANDOM_STATE = 42

FITUR_NUMERIK = [
    "moisture",
    "category_one_defects",
    "quakers",
    "category_two_defects",
    "altitude_low_meters",
    "altitude_high_meters",
    "altitude_mean_meters",
]
FITUR_KATEGORIK = [
    "species",
    "country_of_origin",
    "region",
    "variety",
    "processing_method",
    "color",
]
FITUR = FITUR_NUMERIK + FITUR_KATEGORIK
NAMA_KELAS = ["Di Bawah Ambang", "Kualitas Tinggi"]


def muat_data() -> pd.DataFrame:
    """Muat dataset lokal atau unduh dari sumber publik jika belum tersedia."""
    if not PATH_DATA.exists():
        print("Dataset lokal tidak ditemukan. Mengunduh dataset...")
        with urlopen(URL_DATA, timeout=30) as response:
            PATH_DATA.write_bytes(response.read())

    return pd.read_csv(PATH_DATA)


def buat_pipeline(model) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numerik",
                SimpleImputer(strategy="median"),
                FITUR_NUMERIK,
            ),
            (
                "kategorik",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                FITUR_KATEGORIK,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def main() -> None:
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    data = muat_data()

    kolom_diperlukan = ["total_cup_points", *FITUR]
    kolom_hilang = sorted(set(kolom_diperlukan) - set(data.columns))
    if kolom_hilang:
        raise ValueError(
            "Dataset tidak memiliki kolom yang diperlukan: "
            + ", ".join(kolom_hilang)
        )

    data = data.dropna(subset=["total_cup_points"]).copy()
    data["kualitas_tinggi"] = (
        data["total_cup_points"] >= AMBANG_KUALITAS
    ).astype(int)
    X = data[FITUR]
    y = data["kualitas_tinggi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_specs = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    hasil = []
    prediksi = {}
    for nama, estimator in model_specs.items():
        model = buat_pipeline(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediksi[nama] = y_pred
        hasil.append(
            {
                "model": nama,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    evaluasi = pd.DataFrame(hasil)
    evaluasi.to_csv(DIR_OUTPUT / "evaluasi_model.csv", index=False)

    nama_model_terbaik = evaluasi.sort_values(
        ["f1_score", "accuracy"], ascending=False
    ).iloc[0]["model"]
    y_pred_terbaik = prediksi[nama_model_terbaik]
    hasil_prediksi = pd.DataFrame(
        {
            "kualitas_aktual": [
                NAMA_KELAS[nilai] for nilai in y_test.to_numpy()
            ],
            "kualitas_prediksi": [
                NAMA_KELAS[nilai] for nilai in y_pred_terbaik
            ],
        },
        index=y_test.index,
    ).sort_index()
    hasil_prediksi.to_csv(DIR_OUTPUT / "hasil_prediksi.csv", index_label="index")

    with (DIR_OUTPUT / "laporan_klasifikasi.txt").open(
        "w", encoding="utf-8"
    ) as laporan:
        laporan.write(
            f"Ambang kualitas tinggi: total_cup_points >= {AMBANG_KUALITAS}\n"
        )
        laporan.write(f"Jumlah data: {len(data)}\n")
        laporan.write(f"Jumlah data training: {len(X_train)}\n")
        laporan.write(f"Jumlah data testing: {len(X_test)}\n")
        laporan.write(f"Model terbaik (F1-score): {nama_model_terbaik}\n\n")
        for nama, y_pred in prediksi.items():
            laporan.write(f"=== {nama} ===\n")
            laporan.write(
                classification_report(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                    target_names=NAMA_KELAS,
                    zero_division=0,
                )
            )
            laporan.write("\n")

    ax = evaluasi.set_index("model")[
        ["accuracy", "precision", "recall", "f1_score"]
    ].plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0)
    ax.set_title("Perbandingan Kinerja Model")
    ax.set_ylabel("Skor")
    ax.set_xlabel("Model")
    ax.legend(title="Metrik")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "perbandingan_model.png", dpi=150)
    plt.close()

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_terbaik,
        labels=[0, 1],
        display_labels=NAMA_KELAS,
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(f"Confusion Matrix - {nama_model_terbaik}")
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Dataset: {PATH_DATA}")
    print(f"Data: {len(data)} baris; training: {len(X_train)}; testing: {len(X_test)}")
    print(f"Model terbaik berdasarkan F1-score: {nama_model_terbaik}")
    print(evaluasi.to_string(index=False))
    print(f"Semua hasil tersimpan di: {DIR_OUTPUT}")


if __name__ == "__main__":
    main()