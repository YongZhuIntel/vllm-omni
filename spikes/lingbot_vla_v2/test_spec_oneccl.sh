
I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH
export CCL_PLUGIN=ONECCL_IGPU


cd /llm/zhuyong/lingbovla/my/vllm-omni
bash examples/online_serving/lingbot_vla_v2/run_openvino_comparison.sh \
    --no-prepare \
    --model /tmp/lingbot-vla-v2-perf \
    --spec-decode \
    --accept-rate 0,0.25,0.5,0.75,0.9,1,measured \
    --spec-ticks 40 \
    --spec-worker-cpu 11
