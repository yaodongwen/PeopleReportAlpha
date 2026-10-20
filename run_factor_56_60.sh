#!/bin/bash
set -u

BASE_XML="ai_v2.xml"
ALPHA_MODULE="/home/dwyao/WorkPlace/PeopleDailyFactor/PeopleDailyAIPolicyAlpha_v11.py"
NEW_NPZ="data/factor/people_daily_concentration_novelty_factor.npz"

XML_DIR="xml_factor56_60"
LOG_DIR="logs_factor56_60"

mkdir -p "${XML_DIR}" "${LOG_DIR}"

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: ${BASE_XML} not found"
    exit 1
fi

if [ ! -f "${ALPHA_MODULE}" ]; then
    echo "ERROR: ${ALPHA_MODULE} not found"
    exit 1
fi

if [ ! -f "${NEW_NPZ}" ]; then
    echo "ERROR: ${NEW_NPZ} not found"
    echo "Run first:"
    echo "  cd ai_policy_label"
    echo "  python 07_build_concentration_novelty_factors.py"
    echo "  cd .."
    exit 1
fi

FACTORS=(54 56 57 58 59 60)

echo "============================================================"
echo "PeopleDaily Concentration + First-Appearance Test"
echo "============================================================"
echo "54 = frozen current best"
echo "56 = policy concentration surprise + accel veto"
echo "57 = first/reappearance-after-365d novelty + accel veto"
echo "58 = 0.75*Rank(F54) + 0.25*Rank(F56)"
echo "59 = 0.75*Rank(F54) + 0.25*Rank(F57)"
echo "60 = 0.70*Rank(F54) + 0.15*Rank(F56) + 0.15*Rank(F57)"
echo ""
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "vetoThreshold=-0.6"
echo "============================================================"

for FACTOR_TYPE in "${FACTORS[@]}"
do
    FACTOR_ID="factor${FACTOR_TYPE}-cn"
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
        if [ -e "pnl/${FACTOR_ID}" ]; then
            simsummary "pnl/${FACTOR_ID}"
        else
            echo "WARNING: pnl/${FACTOR_ID} not found"
        fi
    else
        echo "[FAILED] ${FACTOR_ID}"
        echo "log: ${LOG_FILE}"
        tail -120 "${LOG_FILE}"
    fi
done

echo ""
echo "============================================================"
echo "CORRELATIONS TO F54"
echo "============================================================"

if [ -e "pnl/factor54-cn" ]; then
    for F in 56 57 58 59 60
    do
        if [ -e "pnl/factor${F}-cn" ]; then
            printf "corr(F54,F%s) = " "${F}"
            bcorr \
                "pnl/factor54-cn" \
                "pnl/factor${F}-cn" \
                2 2 2>/dev/null
        fi
    done
fi

echo ""
echo "============================================================"
echo "NEW FACTOR CROSS-CORRELATION"
echo "============================================================"

if [ -e "pnl/factor56-cn" ] && [ -e "pnl/factor57-cn" ]; then
    printf "corr(F56,F57) = "
    bcorr pnl/factor56-cn pnl/factor57-cn 2 2 2>/dev/null
fi

echo ""
echo "============================================================"
echo "TEST FINISHED"
echo "============================================================"
