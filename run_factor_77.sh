#!/bin/bash
set -u

BASE_XML="ai.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v20.py"

NEW_NPZ="data/factor/people_daily_industry_specificity_factor.npz"

XML_DIR="xml_factor77"
LOG_DIR="logs_factor77"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${NEW_NPZ}" ]; then
    echo "ERROR: ${NEW_NPZ} not found"
    echo "Run first:"
    echo "  cd ai_policy_label"
    echo "  python 11_build_industry_specificity_factor.py"
    echo "  cd .."
    exit 1
fi

FACTORS=(54 73 77)

echo "============================================================"
echo "PeopleDaily Industry Specificity Standalone Test"
echo "============================================================"
echo "54 = frozen benchmark"
echo "73 = signed consistency low-correlation reference"
echo "77 = industry specificity standalone"
echo ""
echo "Hard requirement:"
echo "  max absolute pool correlation < 0.20"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-specificity"
    XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"
    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"

    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "============================================================"

    sed \
        -e 's#module="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha[^"]*\.py"#module="'"${ALPHA_MODULE}"'"#g' \
        -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        -e 's/extremeCut="[+-]*[0-9.]*"/extremeCut="0.3"/g' \
        -e 's/reactionWindow="[0-9]*"/reactionWindow="5"/g' \
        -e 's/vetoThreshold="[+-]*[0-9.]*"/vetoThreshold="-0.6"/g' \
        -e 's/debug="[0-9]*"/debug="0"/g' \
        "${BASE_XML}" > "${XML_FILE}"

    alphasim "${XML_FILE}" > "${LOG_FILE}" 2>&1
    STATUS=$?

    if [ ${STATUS} -eq 0 ]; then
        echo "[OK] ${FACTOR_ID}"
        simsummary "pnl/${FACTOR_ID}"
    else
        echo "[FAILED] ${FACTOR_ID}"
        echo "log: ${LOG_FILE}"
        tail -160 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "PAIRWISE CORRELATIONS"
echo "============================================================"

for PAIR in "54 73" "54 77" "73 77"
do
    set -- ${PAIR}
    A=$1
    B=$2

    if [ -e "pnl/factor${A}-specificity" ] && [ -e "pnl/factor${B}-specificity" ]; then
        printf "corr(F%s,F%s) = " "${A}" "${B}"
        bcorr \
            "pnl/factor${A}-specificity" \
            "pnl/factor${B}-specificity" \
            2 2 2>/dev/null
    fi
done

PNL_PATH="pnl/factor77-specificity"

if [ -e "${PNL_PATH}" ]; then
    echo ""
    echo "============================================================"
    echo "FACTOR77 RISK CORRELATIONS"
    echo "============================================================"

    bcorr \
        "${PNL_PATH}" \
        /mnt/nas-intern/homes/dwyao/alpha_pnl_risk_sim3/* \
        2 2

    echo ""
    echo "============================================================"
    echo "FACTOR77 POOL EXTREMES"
    echo "============================================================"

    bcorr "${PNL_PATH}" /data/share/poolpnl/* 2 2 2>/dev/null \
    | sort -u -g -k 1 \
    | awk '
    { lines[NR]=$0 }
    END {
        print "--- min ---"
        for(i=1;i<=10 && i<=NR;i++) print lines[i]
        print "--- max ---"
        start=NR-9
        if(start<1) start=1
        for(i=start;i<=NR;i++) print lines[i]
    }'
else
    echo "[SKIP] Factor77 pnl not found."
fi

echo ""
echo "============================================================"
echo "FACTOR77 SPECIFICITY TEST FINISHED"
echo "============================================================"
