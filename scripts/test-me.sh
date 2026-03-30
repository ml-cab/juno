cd ../
echo ===============================
echo "TinyLlama-1.1B-Q5KM.llamafile"
echo ===============================
./juno test --model-path ../models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
echo ===============================
echo "TinyLlama-1.1B-Q4KM.gguf"
echo ===============================
./juno test --model-path ../models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
echo ===============================
echo "Meta-Llama-3.2B-Q8_0.llamafile"
echo ===============================
./juno test --model-path ../models/Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
echo ===============================
echo "phi-3.5-Q4KM.gguf"
echo ===============================
./juno test --model-path ../models/phi-3.5-mini-instruct.Q4_K_M.gguf --heap 4g
