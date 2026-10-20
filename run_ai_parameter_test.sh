#!/bin/bash

set -u


# ============================================================
# Config
# ============================================================

BASE_XML="ai.xml"

XML_DIR="xml_runs"
LOG_DIR="logs"

mkdir -p "${XML_DIR}"
mkdir -p "${LOG_DIR}"


if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: missing template ${BASE_XML}"
    exit 1
fi


# ============================================================
# Factors
# ============================================================

FACTOR_TYPES=(
    17
    20
    21
    22
)


# ============================================================
# Extreme cuts
# ============================================================

EXTREME_CUTS=(
    0.0
    0.3
    0.5
)


# ============================================================
# Signs
# ============================================================

SIGNS=(
    -1
)


# ============================================================
# Start
# ============================================================

echo "============================================================"
echo "PeopleDaily AI Parameter Test"
echo "============================================================"


for FACTOR_TYPE in "${FACTOR_TYPES[@]}"
do

    for SIGN in "${SIGNS[@]}"
    do

        for CUT in "${EXTREME_CUTS[@]}"
        do

            # =================================================
            # ID
            # =================================================

            if [ "${SIGN}" = "-1" ]; then
                SIGN_LABEL="m1"
            else
                SIGN_LABEL="p1"
            fi


            CUT_LABEL=$(echo "${CUT}" | sed 's/\.//g')


            FACTOR_ID="factor${FACTOR_TYPE}-${SIGN_LABEL}-c${CUT_LABEL}"

            XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"

            LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"


            echo ""
            echo "============================================================"
            echo "Running ${FACTOR_ID}"
            echo "factorType=${FACTOR_TYPE}"
            echo "sign=${SIGN}"
            echo "extremeCut=${CUT}"
            echo "============================================================"


            # =================================================
            # Generate XML
            # =================================================

            sed \
                -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
                -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
                -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
                -e 's/sign="[+-]*[0-9.]*"/sign="'"${SIGN}"'"/g' \
                -e 's/extremeCut="[0-9.]*"/extremeCut="'"${CUT}"'"/g' \
                -e 's/debug="[0-9]*"/debug="0"/g' \
                "${BASE_XML}" \
                > "${XML_FILE}"


            # =================================================
            # Run AlphaSim
            # =================================================

            START_TIME=$(date +%s)

            alphasim "${XML_FILE}" \
                > "${LOG_FILE}" \
                2>&1

            STATUS=$?

            END_TIME=$(date +%s)

            ELAPSED=$((END_TIME - START_TIME))


            # =================================================
            # Result
            # =================================================

            if [ ${STATUS} -eq 0 ]; then

                echo ""
                echo "[OK] ${FACTOR_ID}"
                echo "elapsed: ${ELAPSED}s"


                if [ -e "pnl/${FACTOR_ID}" ]; then

                    echo ""
                    simsummary "pnl/${FACTOR_ID}"

                else

                    echo ""
                    echo "WARNING: pnl/${FACTOR_ID} not found"

                    echo ""
                    echo "Possible pnl files:"
                    ls -1 pnl 2>/dev/null | grep "${FACTOR_TYPE}" || true

                fi

            else

                echo ""
                echo "[FAILED] ${FACTOR_ID}"
                echo "exit code: ${STATUS}"
                echo "log: ${LOG_FILE}"

                echo ""
                tail -40 "${LOG_FILE}"

            fi


        done

    done

done


echo ""
echo "============================================================"
echo "ALL TESTS FINISHED"
echo "============================================================"