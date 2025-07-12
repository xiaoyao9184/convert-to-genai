#!/bin/bash

MODEL_ID=$1
SAME_REPO=$2
URL="${3:-http://localhost:7860}"
TIMEOUT=60
INTERVAL=2

echo "⏳ Waiting for Gradio to be ready at $URL ..."

for ((i=1; i<=TIMEOUT/INTERVAL; i++)); do
    if curl -s --fail "$URL" > /dev/null; then
        break
    fi

    if (( i == TIMEOUT / INTERVAL )); then
        echo "❌ Gradio is not ready after $((i * INTERVAL)) seconds!"
        exit 1
    fi
    echo "⏳ Still waiting... ($((i*INTERVAL))s)"
    sleep $INTERVAL
done

EVENT_ID=$(curl -X POST $URL/gradio_api/call/click_proceed \
    -s -H "Content-Type: application/json" -d '{
        "data": [
            "'$MODEL_ID'",
            '$SAME_REPO',
            {}
        ]}' \
    | awk -F'"' '{ print $4 }')

curl -s -N $URL/gradio_api/call/click_proceed/$EVENT_ID \
    | awk '
        /^event:/ {
        event_type = $0;
        next;
        }
        /^data:/ {
        sub(/^data: /, "", $0);
        if (event_type == "event: heartbeat") {
            print "__HEARTBEAT__";
        } else {
            print $0;
        }
        fflush();
        }
    ' \
    | while IFS= read -r line; do
      if [[ "$line" == "__HEARTBEAT__" ]]; then
        printf "░"
      else
        content=$(echo "$line" | jq -r '.[1] // empty')
        if [[ -n "$content" ]]; then
          echo -e "\n─────────────"
          echo "$content"
          echo "─────────────"
        fi
      fi
    done
