"""Descriptive article notes from completed outputs; never trains or selects a model."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.factorial_model import CELLS
from src.factorial_report import LABELS
from src.utils import save_json
from src.visualization import save


def main():
    root = Path("results/factorial")
    status = json.loads((root/"status.json").read_text())
    assert status["status"] == "complete" and not status["quick"]
    comparison = pd.read_csv(root/"tables/comparison.csv").set_index("model")
    contrasts = pd.read_csv(root/"tables/factorial_contrasts.csv")
    selective = pd.read_csv(root/"tables/selective_risk_coverage.csv")
    summary = pd.read_csv(root/"tables/summary.csv")
    primary = contrasts[contrasts.primary]
    protocol = json.loads((root/"protocol.json").read_text())
    cohort = pd.read_csv(root/"tables/development_cohort.csv")
    distributions = []
    for fold in protocol["outer_folds"]:
        for role, ids in fold["participant_ids"].items():
            records = cohort[cohort.subject_id.isin(ids)]
            for field in ["sbp_mmHg", "dbp_mmHg"]:
                distributions.append({"fold": fold["fold"], "role": role, "target": field, "participants": records.subject_id.nunique(),
                                      "records": len(records), "mean": records[field].mean(), "SD": records[field].std(ddof=1),
                                      "minimum": records[field].min(), "median": records[field].median(), "maximum": records[field].max()})
    pd.DataFrame(distributions).to_csv(root/"tables/split_target_distributions.csv", index=False)
    lines = ["# Ringkasan bukti untuk penulisan artikel", "",
             "Ini analisis deskriptif setelah eksperimen faktorial selesai. Tidak mengubah model, ambang, konfigurasi, atau hasil eksperimen; bukan test independen baru.", "",
             f"Eksperimen lengkap: {status['neural_fits']} fit, 3 fold partisipan, 3 seed, {status['development_records']} rekaman / {status['development_subjects']} partisipan. Sebanyak {status['retired_test_subjects']} partisipan test historis tetap dikeluarkan.", "",
             "## Apa yang diperbaiki secara metodologis", "",
             "- Pengaruh gating dibandingkan pada encoder, parameter, quality supervision dan aturan seleksi yang sama: B−A tanpa augmentasi, D−C dengan augmentasi. C−A dan D−B memisahkan pengaruh augmentasi. Semua sel memakai mask sensor yang sama dan ranking loss dinonaktifkan.",
             "- Seluruh checkpoint dan ambang dikunci sebelum evaluasi outer. Baseline mean/median dan scaler hanya fit pada inner training. Manual quality tidak menjadi masukan inference. Tidak ada label BP sintetis atau pengunduhan dataset baru.",
             "- Empat kontras primer dilaporkan dengan interval bootstrap bersyarat yang disesuaikan Bonferroni. Seluruh seed, kondisi, hasil negatif, serta ambang coverage ikut dilaporkan. Interval sekunder bersifat eksploratori.", "",
             "## Akurasi bersih", "", "MAE mmHg, rata-rata seluruh seed; bukan seed terbaik.", "",
             "| Model | SBP MAE | DBP MAE |", "|---|---:|---:|"]
    for name, row in comparison.iterrows():
        lines.append(f"| {LABELS[name]} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} |")
    lines += ["", "## RQ2: bukti gating yang terpisah dari augmentasi", ""]
    for row in primary.itertuples():
        lo, hi = row.primary_family_Bonferroni_lower, row.primary_family_Bonferroni_upper
        evidence = "mendukung penurunan MAE pada analisis development ini" if hi < 0 else ("menunjukkan kenaikan MAE pada analisis development ini" if lo > 0 else "manfaat belum terbukti; bukan bukti ekuivalensi")
        lines.append(f"- {row.contrast}, {row.target}: perubahan MAE {row.delta_MAE:+.3f} mmHg, interval disesuaikan [{lo:+.3f}, {hi:+.3f}]; {evidence}.")
    lines += ["", "## RQ1: multimodal dibanding unimodal", ""]
    for row in contrasts[(contrasts.condition == "clean") & contrasts.contrast.isin(["multimodal_minus_ecg", "multimodal_minus_ppg"])].itertuples():
        lines.append(f"- {row.contrast}, {row.target}: {row.delta_MAE:+.3f} mmHg, interval eksploratori [{row.lower_95:+.3f}, {row.upper_95:+.3f}]. Negatif menguntungkan multimodal; bandingkan A dengan kontrol clean yang sama.")
    a, dense = comparison.loc["A_uniform_clean"], comparison.loc["dense_uniform_clean"]
    lines += ["", "## RQ3/RQ5: lightweight dan trade-off", "",
              f"A memiliki {int(a.parameters):,} parameter, dibanding kontrol dense {int(dense.parameters):,}: pengurangan {100*(1-a.parameters/dense.parameters):.1f}%. MAC turun {100*(1-a.MACs/dense.MACs):.1f}% ({int(a.MACs):,} vs {int(dense.MACs):,}).",
              f"Latency model batch 1 yang benar-benar diukur: A {a.latency_ms:.3f} ms, dense {dense.latency_ms:.3f} ms pada perangkat yang sama. Penghematan parameter/MAC tidak otomatis berarti latency lebih rendah. Error, ukuran model, throughput dan RSS juga tersedia pada tabel lengkap.", "",
              "## Selective prediction: jangan menyembunyikan coverage", ""]
    d = selective[(selective.model == "D_gated_augmented") & (selective.condition == "clean")]
    for coverage, data in d.groupby("validation_target_coverage", sort=False):
        values = data.groupby("target").mean(numeric_only=True)
        lines.append(f"- Target validation {coverage:.0%}; coverage aktual {values.actual_record_coverage.mean():.1%}; MAE diterima SBP/DBP {values.loc['SBP','selective_MAE']:.3f}/{values.loc['DBP','selective_MAE']:.3f} mmHg; random rejection dengan jumlah penerimaan sama {values.loc['SBP','random_rejection_MAE_mean']:.3f}/{values.loc['DBP','random_rejection_MAE_mean']:.3f} mmHg.")
    lines += ["", "Gunakan semua operating point dan `selective_subgroups.csv`. Penurunan error pada sebagian data tidak boleh disebut peningkatan seluruh populasi. Penolakan berbasis kualitas dapat mengurangi coverage dan tetap gagal menurunkan error BP.", "",
              "## RQ4 dan batas klaim", "",
              "Prediksi OOF tidak memakai partisipan yang sama dalam fitting fold-nya. Namun cohort pernah dianalisis dan desain dipengaruhi hasil sebelumnya. Ini bukti development lintas partisipan, bukan konfirmasi eksternal/future holdout. Bootstrap bersyarat tidak memasukkan seluruh ketidakpastian training dan ketergantungan training antar-fold.",
              "Model unimodal memakai satu encoder dan memiliki lebih sedikit parameter; perbandingannya menilai konfigurasi praktis, bukan kontrol kapasitas identik untuk mengisolasi informasi modalitas kedua. Hyperparameter/budget yang disamakan juga tidak membuktikan bahwa setiap arsitektur sudah mencapai performa terbaiknya.",
              "Kualitas manual visual tidak identik dengan kegunaan sinyal untuk BP. Cuff reference per-record, jumlah partisipan, ketidaksesuaian paket dataset, partial quality-window overlap, dan representasi amplitudo digital membatasi interpretasi. Hasil ini bukan validasi perangkat medis.", "",
              "## Posisi artikel dan pemeriksaan literatur terbatas", "",
              "Pencarian sumber primer ini bukan systematic review atau bukti kebaruan. Dua preprint berikut menunjukkan bahwa sekadar menambahkan kualitas/gating pada ECG–PPG tidak cukup untuk mengklaim metode pertama:",
              "- [QualityFM](https://arxiv.org/abs/2509.06516) (preprint 2025) mempelajari kualitas ECG/PPG dengan self-distillation dan mengevaluasi transfer ke estimasi ABP. Skala data/model dan protokolnya berbeda; tidak ada perbandingan angka MAE lintas studi di sini.",
              "- [CardioFusion-AI](https://arxiv.org/abs/2608.26000) (preprint 2026; halaman menyatakan under review) membandingkan strategi ECG–PPG pada degradasi sintetis dan membedakan sensor tersedia dari kualitas sensor. Metrik tugas sintetik yang disajikan adalah heart rate dalam bpm, bukan SBP/DBP dalam mmHg; bukan baseline akurasi BP yang sebanding.",
              "- [Rekomendasi evaluasi algoritma BP](https://pmc.ncbi.nlm.nih.gov/articles/PMC11245506/) mendukung pelaporan error/agreement serta keterbatasan benchmark secara transparan.",
              "- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) menjadi latar selective prediction; kebijakan quantile kualitas di sini bukan arsitektur SelectiveNet. [Selective Regression under Fairness Criteria](https://proceedings.mlr.press/v162/shah22a.html) menjelaskan mengapa subgroup coverage/error perlu diperiksa.", "",
              "Kontribusi yang dapat dikembangkan adalah evaluasi terkontrol yang reproducible pada dataset ini: memisahkan gating, augmentasi, ketersediaan sensor, baseline populasi, dan biaya komputasi. Keunggulan kualitas/akurasi hanya boleh diklaim pada kondisi yang didukung hasil; metode yang terimplementasi tidak otomatis unggul atau baru.",
              "Alternatif judul deskriptif: **A Controlled Evaluation of Quality-Aware Lightweight ECG–PPG Fusion for Cuffless Blood Pressure Estimation**. Judul awal dapat dipakai secara deskriptif jika isi/abstrak tidak mengklaim peningkatan yang tidak terbukti.", "",
              "Langkah berikut harus berupa hipotesis baru yang dinyatakan sebelum validasi baru, misalnya task-specific reliability dengan target residual out-of-fold pada training saja. Jangan terus memilih arsitektur/ambang pada cohort yang sama lalu menyebutnya test baru. Tidak ada jaminan acceptance jurnal dari validitas pipeline saja.", ""]
    (root/"ARTICLE_NOTES_ID.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ["SBP", "DBP"], strict=True):
        for name in CELLS:
            table = selective[(selective.model == name) & (selective.condition == "clean") & (selective.target == target)].groupby("validation_target_coverage").mean(numeric_only=True)
            line, = ax.plot(table.actual_record_coverage*100, table.selective_MAE, "o-", label=LABELS[name])
            ax.plot(table.actual_record_coverage*100, table.random_rejection_MAE_mean, "--", color=line.get_color(), alpha=.6)
        ax.set(xlabel="Actual retained records (%)", ylabel=f"{target} MAE (mmHg)", title="Solid=quality rejection; matching dashed=random rejection")
        ax.legend(fontsize=7)
    save(fig, root/"figures", "selective_risk_coverage_matched_colors")
    snapshot = root/"postprocessing_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    save_json(root/"postprocessing_manifest.json", {"scope": "descriptive Indonesian article notes and clarified plot colors; no model/threshold selection",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "table_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"tables").glob("*.csv")},
              "sources_checked": ["https://arxiv.org/abs/2509.06516", "https://arxiv.org/abs/2608.26000",
                                  "https://pmc.ncbi.nlm.nih.gov/articles/PMC11245506/", "https://proceedings.mlr.press/v97/geifman19a.html",
                                  "https://proceedings.mlr.press/v162/shah22a.html"], "literature_scope": "bounded primary-source scan, not a systematic novelty review"})


if __name__ == "__main__":
    main()
