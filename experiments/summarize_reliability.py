"""Post-hoc descriptive article notes; never changes fitted policies or thresholds."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.reliability_report import LABELS
from src.utils import save_json


def conclusion(lower,upper):
    if upper<0:
        return "mendukung penurunan error secara bersyarat pada studi development ini"
    if lower>0:
        return "menunjukkan kenaikan error secara bersyarat pada studi development ini"
    return "manfaat belum terbukti; bukan bukti ekuivalensi"


def main():
    root=Path("results/reliability")
    status=json.loads((root/"status.json").read_text())
    assert status["status"]=="complete" and not status["quick"]
    protocol=json.loads((root/"protocol.json").read_text())
    config=protocol["config"]
    comparison=pd.read_csv(root/"tables/comparison.csv")
    contrasts=pd.read_csv(root/"tables/fusion_contrasts.csv")
    diagnostic=pd.read_csv(root/"tables/feature_gate_contrasts.csv")
    selective=pd.read_csv(root/"tables/selective_risk_coverage.csv")
    subgroups=pd.read_csv(root/"tables/selective_subgroups.csv")
    bounds=[]
    for seed in config["seeds"]:
        frame=pd.read_csv(root/"predictions"/f"adaptive_risk_fusion_seed_{seed}_clean.csv")
        for target in ["sbp","dbp"]:
            y=frame[target+"_mmHg"].to_numpy()
            experts=frame[[f"ecg_{target}_expert_prediction",f"ppg_{target}_expert_prediction"]].to_numpy()
            best_convex=np.clip(y,experts.min(1),experts.max(1))
            current=np.abs(frame[target+"_predicted"].to_numpy()-y)
            lower=np.abs(best_convex-y)
            residuals=experts-y[:,None]
            assert (lower<=current+1e-4).all()
            bounds.append({"seed":seed,"target":target.upper(),"current_MAE":current.mean(),"label_dependent_convex_oracle_MAE":lower.mean(),
                           "maximum_remaining_MAE_reduction_for_these_experts":(current-lower).mean(),
                           "expert_signed_error_correlation":np.corrcoef(residuals.T)[0,1],
                           "reference_outside_expert_prediction_interval_fraction":float(((y<experts.min(1))|(y>experts.max(1))).mean()),
                           "scope":"post-hoc diagnostic uses true outer BP; never an inference method or training target; not manual-quality oracle"})
    pd.DataFrame(bounds).to_csv(root/"tables/descriptive_convex_fusion_headroom.csv",index=False)
    lines=["# Hasil perbaikan quality-aware dan reliabilitas BP", "",
           "Semua angka berasal dari eksperimen selesai. Ini studi development post-hoc pada cohort yang pernah dianalisis, bukan test konfirmatori baru.","",
           f"Cohort: {status['development_subjects']} partisipan / {status['development_records']} rekaman, tiga outer fold, tiga seed. {status['retired_test_subjects']} partisipan test historis tetap dikeluarkan. Eksekusi baru: {status['new_teacher_fits']} teacher neural, {status['risk_head_fits']} risk head, {status['outer_policy_evaluations']} evaluasi kebijakan outer.","",
           "## Perbaikan yang benar-benar diuji", "",
           "1. Uji gate acak, override uniform, dan oracle manual pada model feature-fusion lama yang dibekukan; prediksi learned gate direproduksi terhadap hasil tersimpan.",
           "2. Oracle manual tambahan pada late fusion. Seluruh oracle manual memakai rekaman asli/clean, nondeployable, dan bukan batas performa yang dijamin.",
           "3. Risk head mempelajari error SBP/DBP per modalitas dari residual participant-out-of-fold di dalam inner training saja. Rekaman target residual tidak muncul pada training atau validation teacher pembentuk prediksinya.",
           "4. Bobot fusion khusus SBP/DBP berdasarkan risiko; lima kebijakan memakai expert yang sama. BP-only risk menjadi kontrol tambahan agar kontribusi SQI/kualitas dapat dinilai terpisah dari informasi prediksi BP.",
           "5. Threshold retention dari validation saja, seluruh coverage dan subkelompok tetap dilaporkan. Tidak ada pemilihan model dari hasil outer atau label cuff sintetis.","",
           "## Akurasi pada rekaman asli", "", "MAE mmHg, rata-rata tiga seed; bukan seed terbaik atau ensemble prediksi.","",
           "| Kebijakan | SBP MAE | DBP MAE |", "|---|---:|---:|"]
    for row in comparison.itertuples():
        lines.append(f"| {LABELS[row.model]} | {row.SBP_MAE:.3f} | {row.DBP_MAE:.3f} |")
    lines += ["", "Arsitektur ini melakukan late fusion atas dua expert BP yang dibekukan. Hasilnya tidak boleh digunakan untuk mengubah angka atau mengatribusikan seluruh perubahan terhadap quality gating pada arsitektur feature-fusion lama.","",
              "## Bukti primer kontribusi informasi kualitas", "",
              "Kontras primer adalah adaptive lengkap dikurangi BP-only risk, pada clean dan rata-rata lima gangguan, masing-masing SBP/DBP. Interval bootstrap partisipan disesuaikan Bonferroni atas empat kontras (98,75% masing-masing); seluruh interval bersyarat pada prediksi yang sudah difit.",""]
    for row in contrasts[contrasts.primary].itertuples():
        lines.append(f"- {row.condition}, {row.target}: delta MAE {row.delta_MAE:+.5f} mmHg, interval [{row.adjusted_lower:+.5f}, {row.adjusted_upper:+.5f}]; {conclusion(row.adjusted_lower,row.adjusted_upper)}.")
    lines += ["", "Besarnya pengurangan clean hanya sekitar 0,013 mmHg untuk SBP dan 0,002 mmHg untuk DBP. Interval bersyarat yang tidak mencakup nol tidak otomatis membuktikan manfaat praktis atau klinis. Kontrol BP-only mempunyai lebih sedikit koefisien aktif, meskipun ukuran layer checkpoint disamakan; kontrol fitur acak dengan jumlah koefisien aktif sama belum diuji. Klaim yang didukung hanya tambahan kecil dari kumpulan input kualitas pada risk head dan cohort ini, bukan bukti semua mekanisme quality-aware unggul."]
    lines += ["", "## Diagnosis mekanisme gate lama", "",
              "Delta di bawah adalah intervensi dikurangi learned gate. Semua interval 95% eksploratori. Model tidak dilatih ulang setelah gate diganti, sehingga perubahan distribusi representasi juga dapat memengaruhi hasil.",""]
    for row in diagnostic.itertuples():
        lines.append(f"- {row.comparison}, {row.target}: {row.delta_MAE:+.3f} mmHg, interval [{row.lower_95:+.3f}, {row.upper_95:+.3f}].")
    lines += ["", "Shuffle yang memperburuk hasil mendukung sensitivitas terhadap pemetaan skor, tetapi tidak sendirian membuktikan keunggulan terhadap model tanpa gating yang dilatih sebanding. Oracle manual yang tidak membantu juga tidak membuktikan semua pendekatan kualitas gagal; anotasi visual, crop, dan bentuk fusion dapat membatasi hasil.","",
              "## Perbandingan fusion sekunder", ""]
    for row in contrasts[(contrasts.condition=="clean") & ~contrasts.primary].itertuples():
        lines.append(f"- {row.comparison}, {row.target}: {row.delta_MAE:+.3f} mmHg, interval eksploratori [{row.lower_95:+.3f}, {row.upper_95:+.3f}].")
    lines += ["", "## Risiko absolut dan coverage", "",
              "Risk head menghasilkan perkiraan error dalam mmHg; skor kualitas visual tetap terpisah. Rata-rata tertimbang estimasi risiko adalah proxy error fusion, bukan error yang diketahui atau batas ketidakpastian terkalibrasi. Inverse-risk weighting merupakan heuristik tetap, tidak menjamin bobot optimal dan tidak memodelkan korelasi error kedua expert secara eksplisit.",""]
    for selector in ["risk","quality"]:
        for coverage in config["selective_coverages"]:
            data=selective[(selective.condition=="clean")&(selective.selector==selector)&(selective.validation_target_coverage==coverage)].groupby("target").mean(numeric_only=True)
            lines.append(f"- {selector}, target {coverage:.0%}: coverage aktual {data.actual_record_coverage.mean():.1%}; MAE diterima SBP/DBP {data.loc['SBP','selective_MAE']:.3f}/{data.loc['DBP','selective_MAE']:.3f} mmHg; random rejection dengan jumlah penerimaan sama {data.loc['SBP','random_rejection_MAE_mean']:.3f}/{data.loc['DBP','random_rejection_MAE_mean']:.3f} mmHg.")
    lines += ["", "Contoh audit coverage pada target validation 50% (deskriptif; bukan pemilihan operating point):",""]
    for selector in ["risk","quality"]:
        for group in ["SBP_>=140","DBP_>=90","age_>=60"]:
            data=subgroups[(subgroups.selector==selector)&(subgroups.validation_target_coverage==.5)&(subgroups.target=="SBP")&(subgroups.subgroup==group)]
            if len(data):
                lines.append(f"- {selector}, {group}: rata-rata coverage subgroup {data.subgroup_coverage.mean():.1%}.")
    lines += ["", "Penurunan MAE pada data yang diterima tidak berarti akurasi seluruh populasi meningkat. Coverage/error seluruh subkelompok dan semua kondisi tetap tersedia di CSV; random-rejection Monte Carlo SD bukan confidence interval partisipan.","",
              "## Batas kemampuan fusion dari expert yang tersedia", "",
              "Analisis berikut tambahan deskriptif setelah model selesai; menggunakan BP outer asli hanya untuk menghitung batas bawah matematis error pada kombinasi konveks dari dua prediksi expert yang sudah tetap. Ini **oracle BP yang bergantung label**, berbeda dari oracle kualitas manual, tidak deployable, dan tidak digunakan untuk training, pemilihan bobot, atau threshold.",""]
    for target,data in pd.DataFrame(bounds).groupby("target"):
        lines.append(f"- {target}: MAE saat ini {data.current_MAE.mean():.3f}; batas bawah convex-oracle {data.label_dependent_convex_oracle_MAE.mean():.3f}; maksimum pengurangan MAE yang masih mungkin untuk prediksi expert ini {data.maximum_remaining_MAE_reduction_for_these_experts.mean():.3f} mmHg. Korelasi signed error expert {data.expert_signed_error_correlation.mean():.3f}; referensi berada di luar rentang kedua prediksi pada {data.reference_outside_expert_prediction_interval_fraction.mean():.1%} rekaman.")
    lines += ["", "Batas ini menjelaskan ruang perbaikan fusion yang tersedia, bukan jaminan estimator otomatis dapat mencapainya. Jika kedua expert mempunyai error searah yang kuat, perbaikan informasi waveform atau expert BP juga perlu diuji.","",
              "## Batas validitas dan klaim artikel", "",
              "Model memakai data partisipan terpisah untuk setiap evaluasi outer, tetapi cohort sebelumnya sudah dianalisis. Source freeze, kontrol sebanding dan QA memperkuat keterlacakan; semuanya tidak menggantikan validasi pada partisipan/perangkat baru.",
              "Risk head difit pada teacher dengan training pool lebih kecil daripada expert final sehingga kalibrasinya dapat bergeser. Dataset berukuran terbatas, kualitas manual sebagian tidak selaras dengan crop, cuff reference hanya per-record, dan amplitudo digital tidak terkalibrasi secara fisiologis. Gangguan sintetis tidak menggantikan artefak sensor nyata.",
              "Kontribusi harus mengikuti hasil: diagnosis mekanisme, pemisahan kualitas visual dari risiko tugas, kontrol informasi kualitas, efisiensi terukur, serta batas generalisasi. Jangan menyatakan seluruh komponen unggul hanya karena model berjalan atau satu perbandingan eksploratori tampak lebih baik. Kebaruan dan acceptance jurnal belum dijamin.",
              "Audit literatur menyeluruh, cohort eksternal/future holdout, artefak sensor nyata, dan validasi klinis belum dilakukan. Tidak ada dataset tambahan yang diunduh.",""]
    (root/"ARTICLE_NOTES_ID.md").write_text("\n".join(lines))
    snapshot=root/"postprocessing_snapshot"
    snapshot.mkdir(exist_ok=True)
    report=root/"RESEARCH_REPORT.md"
    original=report.read_text()
    if not (snapshot/"RESEARCH_REPORT_original.md").exists():
        (snapshot/"RESEARCH_REPORT_original.md").write_text(original)
    rendered=original.splitlines()
    for row in contrasts[contrasts.primary].itertuples():
        prefix=f"- {row.condition}, {row.target}: "
        for i,line in enumerate(rendered):
            if line.startswith(prefix):
                finding=line.split("; ",1)[1]
                rendered[i]=f"{prefix}{row.delta_MAE:+.5f} mmHg, adjusted interval [{row.adjusted_lower:+.5f}, {row.adjusted_upper:+.5f}]; {finding}"
    report.write_text("\n".join(rendered)+"\n")
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    save_json(root/"postprocessing_manifest.json",{"scope":"descriptive article interpretation, primary-interval precision rendering, and label-dependent convex-fusion headroom; no changes to fits, policies or thresholds",
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "table_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"tables").glob("*.csv")}})


if __name__=="__main__":
    main()
