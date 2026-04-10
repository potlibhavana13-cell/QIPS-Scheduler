package org.qips.rest;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.qips.model.JobMetadata;
import org.qips.model.ScheduleResult;

import java.net.URI;
import java.net.http.*;
import java.util.*;

/**
 * QpsoRestClient — calls the Python QPSO FastAPI service.
 *
 * POST /optimize  → returns optimized job order + metrics
 */
public class QpsoRestClient {

    private final String optimizeUrl;
    private final HttpClient http;
    private final ObjectMapper mapper;

    public QpsoRestClient(String baseUrl) {
        this.optimizeUrl = baseUrl.replaceAll("/$", "");
        this.http        = HttpClient.newBuilder()
            .connectTimeout(java.time.Duration.ofSeconds(5))
            .build();
        this.mapper = new ObjectMapper();
    }

    /**
     * Send a scheduling request to the QPSO Python service.
     *
     * @param jobs      pending job metadata
     * @param nodes     node availability info
     * @param weights   fitness function weights [w1..w6]
     * @return          ScheduleResult with ordered job IDs and metrics
     */
    public ScheduleResult optimize(
            List<JobMetadata> jobs,
            List<Map<String, Object>> nodes,
            double[] weights) throws Exception {

        // Build request payload matching the FastAPI ScheduleRequest schema
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("jobs",           buildJobPayload(jobs));
        payload.put("nodes",          nodes);
        payload.put("weights",        doubleArrayToList(weights));
        payload.put("n_particles",    30);
        payload.put("max_iterations", 100);
        payload.put("beta",           0.75);

        String body = mapper.writeValueAsString(payload);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(optimizeUrl + "/optimize"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .timeout(java.time.Duration.ofSeconds(30))
            .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("QPSO service returned HTTP " + response.statusCode()
                + ": " + response.body());
        }

        Map<?, ?> result = mapper.readValue(response.body(), Map.class);
        return parseResult(result);
    }

    private List<Map<String, Object>> buildJobPayload(List<JobMetadata> jobs) {
        List<Map<String, Object>> list = new ArrayList<>();
        for (JobMetadata j : jobs) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id",                j.getId());
            m.put("name",              j.getName());
            m.put("priority",          j.getPriority());
            m.put("estimated_runtime", j.getEstimatedRuntime());
            m.put("cpu_demand",        j.getCpuDemand());
            m.put("mem_demand",        j.getMemDemand());
            m.put("deadline_slack",    j.getDeadlineSlack());
            m.put("input_size_gb",     j.getInputSizeGb());
            list.add(m);
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    private ScheduleResult parseResult(Map<?, ?> raw) {
        List<String> orderedIds = (List<String>) raw.get("ordered_job_ids");
        double fitness          = ((Number) raw.get("final_fitness")).doubleValue();
        Map<?, ?> metricsRaw    = (Map<?, ?>) raw.get("metrics");
        Map<String, Object> metrics = new HashMap<>();
        if (metricsRaw != null) metricsRaw.forEach((k, v) -> metrics.put(k.toString(), v));
        return new ScheduleResult(orderedIds, fitness, metrics);
    }

    private List<Double> doubleArrayToList(double[] arr) {
        List<Double> list = new ArrayList<>();
        for (double d : arr) list.add(d);
        return list;
    }
}
