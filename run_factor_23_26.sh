#!/bin/bash

set -u

BASE_XML="ai_v2.xml"

XML_DIR="xml_reaction_runs"
LOG_DIR="logs_reaction"

mkdir -p "${XML_DIR}"
mkdir -p "${LOG_DIR}"


if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: missing ${BASE_XML}"
    exit 1
fi


# ============================================================
# Factor23-26
# ============================================================

FACTOR_TYPES=(
    23
    24
    25
    26
)


# ============================================================
# label|priceExtreme|boost|penalty
#
# confirm:
#   only strengthen confirmed states
#
# balanced:
#   strengthen confirmed states
#   weaken conflicting states
# ============================================================

PROFILES=(

    "confirm|0.6|0.50|0.00"

    "balanced|0.6|0.50|0.50"
)


echo "============================================================"
echo "PeopleDaily Market Reaction Factors"
echo "Factor23-26"
echo "sign=-1"
echo "extremeCut=0.3"
echo "reactionWindow=5"
echo "============================================================"


for FACTOR_TYPE in "${FACTOR_TYPES[@]}"
do

    for PROFILE in "${PROFILES[@]}"
    do

        IFS='|' read \
            LABEL \
            PRICE_EXTREME \
            BOOST \
            PENALTY \
            <<< "${PROFILE}"


        FACTOR_ID="factor${FACTOR_TYPE}-${LABEL}"

        XML_FILE="${XML_DIR}/${FACTOR_ID}.xml"

        LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"


        echo ""
        echo "============================================================"
        echo "Running ${FACTOR_ID}"
        echo "factorType=${FACTOR_TYPE}"
        echo "priceExtreme=${PRICE_EXTREME}"
        echo "boost=${BOOST}"
        echo "penalty=${PENALTY}"
        echo "============================================================"


        sed \
            -e 's/id="factor[0-9A-Za-z+_.-]*"/id="'"${FACTOR_ID}"'"/g' \
            -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
            -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
            -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
            -e 's/extremeCut="[0-9.]*"/extremeCut="0.3"/g' \
            -e 's/reactionWindow="[0-9]*"/reactionWindow="5"/g' \
            -e 's/priceExtreme="[0-9.]*"/priceExtreme="'"${PRICE_EXTREME}"'"/g' \
            -e 's/confirmBoost="[0-9.]*"/confirmBoost="'"${BOOST}"'"/g' \
            -e 's/conflictPenalty="[0-9.]*"/conflictPenalty="'"${PENALTY}"'"/g' \
            -e 's/debug="[0-9]*"/debug="0"/g' \
            "${BASE_XML}" \
            > "${XML_FILE}"


        START_TIME=$(date +%s)


        alphasim "${XML_FILE}" \
            > "${LOG_FILE}" \
            2>&1


        STATUS=$?


        END_TIME=$(date +%s)

        ELAPSED=$((END_TIME - START_TIME))


        if [ ${STATUS} -eq 0 ]; then

            echo ""
            echo "[OK] ${FACTOR_ID}"
            echo "elapsed: ${ELAPSED}s"

            if [ -e "pnl/${FACTOR_ID}" ]; then

                echo ""
                simsummary \
                    "pnl/${FACTOR_ID}"

            else

                echo ""
                echo \
                    "WARNING: pnl/${FACTOR_ID} not found"

            fi

        else

            echo ""
            echo "[FAILED] ${FACTOR_ID}"
            echo "exit code: ${STATUS}"
            echo "log: ${LOG_FILE}"

            tail -40 \
                "${LOG_FILE}"

        fi

    done

done


echo ""
echo "============================================================"
echo "ALL FACTOR23-26 TESTS FINISHED"
echo "============================================================"