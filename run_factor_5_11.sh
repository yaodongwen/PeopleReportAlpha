#!/bin/bash

set -u

# ============================================================
# 配置
# ============================================================

BASE_XML="ai.xml"
OUT_DIR="xml_runs"
LOG_DIR="logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# ============================================================
# 检查
# ============================================================

if [ ! -f "${BASE_XML}" ]; then
    echo "ERROR: 找不到模板 XML: ${BASE_XML}"
    exit 1
fi

echo "============================================================"
echo "PeopleDaily AI Factor Batch Run"
echo "template: ${BASE_XML}"
echo "factorType: 5 -> 11"
echo "sign: -1"
echo "============================================================"
echo ""

# ============================================================
# 批量运行
# ============================================================

for FACTOR_TYPE in $(seq 5 11)
do

    FACTOR_ID="factor${FACTOR_TYPE}-1"

    XML_FILE="${OUT_DIR}/${FACTOR_ID}.xml"

    LOG_FILE="${LOG_DIR}/${FACTOR_ID}.log"

    echo ""
    echo "============================================================"
    echo "Running ${FACTOR_ID}"
    echo "factorType=${FACTOR_TYPE}"
    echo "sign=-1"
    echo "XML=${XML_FILE}"
    echo "LOG=${LOG_FILE}"
    echo "============================================================"

    # --------------------------------------------------------
    # 从模板生成新的 XML
    #
    # 1. 修改 Alpha id
    # 2. 修改 factorType
    # 3. sign 保持 -1
    # --------------------------------------------------------

    sed \
        -e 's/id="factor[0-9][0-9]*-1"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/id="test\.valid"/id="'"${FACTOR_ID}"'"/g' \
        -e 's/factorType="[0-9][0-9]*"/factorType="'"${FACTOR_TYPE}"'"/g' \
        -e 's/sign="[+-]*[0-9.]*"/sign="-1"/g' \
        "${BASE_XML}" \
        > "${XML_FILE}"

    # --------------------------------------------------------
    # 运行 AlphaSim
    # --------------------------------------------------------

    START_TIME=$(date +%s)

    alphasim "${XML_FILE}" \
        > "${LOG_FILE}" \
        2>&1

    STATUS=$?

    END_TIME=$(date +%s)

    ELAPSED=$((END_TIME - START_TIME))

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    if [ ${STATUS} -eq 0 ]; then

        echo ""
        echo "[OK] ${FACTOR_ID}"
        echo "elapsed: ${ELAPSED}s"

        if [ -e "pnl/${FACTOR_ID}" ]; then

            echo ""
            echo "---------------- simsummary ----------------"

            simsummary "pnl/${FACTOR_ID}"

            echo "--------------------------------------------"

        else

            echo "[WARNING] 找不到 pnl/${FACTOR_ID}"

        fi

    else

        echo ""
        echo "[FAILED] ${FACTOR_ID}"
        echo "exit code: ${STATUS}"
        echo "log: ${LOG_FILE}"

        # 显示最后30行错误
        tail -30 "${LOG_FILE}"

    fi

done

echo ""
echo "============================================================"
echo "All factor tests finished."
echo "============================================================"