import numpy as np
import pandas as pd

if __name__ == "__main__":
    nopp_gad_H8_L8_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_gad/AD512_H8_L8_coef1e-05/fold_results.pkl"
    nopp_gad_H8_L8_C106 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_gad/AD512_H8_L8_coef1e-06/fold_results.pkl"
    nopp_gad_H16_L4_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_gad/AD512_H16_L4_coef1e-05/fold_results.pkl"
    nopp_gad_H16_L6_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_gad/AD512_H16_L6_coef1e-05/fold_results.pkl"
    nopp_gad_H16_L8_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_gad/AD512_H16_L8_coef1e-05/fold_results.pkl"
    nopp_nogad_H16_L4_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/nopp_nogad/AD512_H16_L4_coef1e-05/fold_results.pkl"
    plo_gad_H16_L4_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/plo_gad/AD512_H16_L4_coef1e-05/fold_results.pkl"
    plo_gad_H8_L8_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/plo_gad/AD512_H8_L8_coef1e-05/fold_results.pkl"
    plo_gad_H16_L6_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/plo_gad/AD512_H16_L6_coef1e-05/fold_results.pkl"
    pur_gad_H16_L4_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/pur_gad/AD512_H16_L4_coef1e-05/fold_results.pkl"
    pp_gad_H16_L4_C105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/pp_gad/AD512_H16_L4_coef1e-05/fold_results.pkl"
    pp_gad_H8_L8_c105 = "/home/asohn3/baraslab/germline_somatic/Results/survival/pp_gad/AD512_H8_L8_coef1e-05/fold_results.pkl"

    fps = [
        nopp_nogad_H16_L4_C105,
        nopp_gad_H16_L4_C105,
        nopp_gad_H8_L8_C105,
        nopp_gad_H8_L8_C106,
        nopp_gad_H16_L6_C105,
        nopp_gad_H16_L8_C105,
        pur_gad_H16_L4_C105,
        plo_gad_H8_L8_C105,
        plo_gad_H16_L4_C105,
        plo_gad_H16_L6_C105,
        pp_gad_H16_L4_C105,
        pp_gad_H8_L8_c105,
    ]

    for fp in fps:
        result = pd.read_pickle(fp)
        ci = []
        auc = []
        for i in range(1, 7):
            fold = result[f"fold_{i}"]
            ci.append(fold["concordance_index"][0])
            auc.append(fold["auroc"][-1])

        ci = np.asarray(ci)
        auc = np.asarray(auc)

        fp_splits = fp.split("/")
        it, mt = fp_splits[-3], fp_splits[-2]
        print(f"{it}, {mt} c-index/time-dynamic-auc: {ci.mean()}/{auc.mean()}")
