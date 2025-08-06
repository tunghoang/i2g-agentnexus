#curl http://127.0.0.1:7000/tools/plot_logplot -X POST --data '{"input": "interpretation/25.las"}' -H "Content-Type: application/json"
#curl http://127.0.0.1:7000/tools/plot_las -X POST --data '{"input": "interpretation/25.las"}' -H "Content-Type: application/json"
curl http://127.0.0.1:7000/tools/summarize_marker_data -X POST --data '{"input": "{}"}' -H "Content-Type: application/json"
