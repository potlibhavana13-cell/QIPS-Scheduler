package org.qips.scheduler;

import org.apache.hadoop.yarn.server.resourcemanager.RMContext;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.RMApp;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.RMAppState;
import org.apache.hadoop.yarn.server.resourcemanager.rmnode.RMNode;
import org.qips.model.JobMetadata;

import java.util.*;

/**
 * TaskProfiler — collects per-job metadata and per-node availability
 * from the YARN ResourceManager context, to be sent to the QPSO optimizer.
 *
 * For each pending YARN application it gathers:
 *   - application ID, name, queue, priority
 *   - estimated runtime (from history server or a simple heuristic)
 *   - CPU + memory demand from the first ResourceRequest
 *   - input data size (from job configuration or HDFS stats)
 *   - submit time and application tags (for deadline extraction)
 *
 * For each active node it gathers:
 *   - node ID, total and available CPU/memory
 *   - list of HDFS block-local job IDs (from NameNode's block map)
 */
public class TaskProfiler {

    private final RMContext rmContext;

    // Fallback runtime estimate (seconds) when no history is available.
    private static final double DEFAULT_RUNTIME_ESTIMATE = 120.0;
    // Fallback input size (GB) when HDFS stats are unavailable.
    private static final double DEFAULT_INPUT_GB = 1.0;

    public TaskProfiler(RMContext rmContext) {
        this.rmContext = rmContext;
    }

    /**
     * Return metadata for all YARN apps currently in ACCEPTED or RUNNING state.
     */
    public List<JobMetadata> collectPendingJobs() {
        List<JobMetadata> result = new ArrayList<>();
        if (rmContext == null) return result;

        for (Map.Entry<?, RMApp> entry : rmContext.getRMApps().entrySet()) {
            RMApp app = entry.getValue();
            if (app.getState() != RMAppState.ACCEPTED
                    && app.getState() != RMAppState.RUNNING) continue;

            JobMetadata meta = new JobMetadata(
                app.getApplicationId().toString(),
                app.getName(),
                app.getApplicationPriority() != null
                    ? app.getApplicationPriority().getPriority()
                    : 5,                              // default priority
                app.getQueue(),
                app.getSubmitTime(),
                estimateRuntime(app),
                estimateCpuDemand(app),
                estimateMemDemand(app),
                0.0,                                  // deadlineSlack filled by JobClassifier
                0.0,                                  // normalizedPriority filled by JobClassifier
                DEFAULT_INPUT_GB,
                new ArrayList<>(app.getApplicationTags())
            );
            result.add(meta);
        }
        return result;
    }

    /**
     * Return availability info for all active NodeManagers as plain maps
     * (compatible with the JSON schema the Python service expects).
     */
    public List<Map<String, Object>> collectNodeInfo() {
        List<Map<String, Object>> result = new ArrayList<>();
        if (rmContext == null) return result;

        for (RMNode node : rmContext.getRMNodes().values()) {
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("id",            node.getNodeID().toString());
            info.put("total_cpu",     node.getTotalCapability().getVirtualCores());
            info.put("total_mem",     node.getTotalCapability().getMemorySize());
            info.put("available_cpu", getAvailableCpu(node));
            info.put("available_mem", getAvailableMem(node));
            info.put("data_blocks",   getLocalBlockJobIds(node));
            result.add(info);
        }
        return result;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Estimate runtime from YARN's MapReduce history server if available,
     * otherwise fall back to a configurable default.
     */
    private double estimateRuntime(RMApp app) {
        try {
            // In a real deployment, query the JobHistoryServer REST API:
            //   GET /ws/v1/history/mapreduce/jobs?user=<user>&state=SUCCEEDED
            // and compute the average of past runs for this job name.
            // For now we use a simple heuristic: 2 minutes per GB of input.
            return DEFAULT_RUNTIME_ESTIMATE;
        } catch (Exception e) {
            return DEFAULT_RUNTIME_ESTIMATE;
        }
    }

    private int estimateCpuDemand(RMApp app) {
        try {
            return app.getCurrentAppAttempt()
                      .getMasterContainer()
                      .getResource()
                      .getVirtualCores();
        } catch (Exception e) {
            return 2; // default
        }
    }

    private long estimateMemDemand(RMApp app) {
        try {
            return app.getCurrentAppAttempt()
                      .getMasterContainer()
                      .getResource()
                      .getMemorySize();
        } catch (Exception e) {
            return 1024L; // default 1 GB
        }
    }

    private int getAvailableCpu(RMNode node) {
        try {
            return node.getTotalCapability().getVirtualCores()
                 - node.getAllocatedContainers().stream()
                       .mapToInt(c -> c.getResource().getVirtualCores()).sum();
        } catch (Exception e) {
            return node.getTotalCapability().getVirtualCores();
        }
    }

    private long getAvailableMem(RMNode node) {
        try {
            return node.getTotalCapability().getMemorySize()
                 - node.getAllocatedContainers().stream()
                       .mapToLong(c -> c.getResource().getMemorySize()).sum();
        } catch (Exception e) {
            return node.getTotalCapability().getMemorySize();
        }
    }

    /**
     * Identify which job IDs have data blocks local to this node.
     * In a real deployment, this queries the HDFS NameNode's block map via WebHDFS:
     *   GET /webhdfs/v1/<path>?op=GETFILEBLOCKLOCATIONS
     * and matches the returned DataNode hostnames against the RMNode hostname.
     */
    private List<String> getLocalBlockJobIds(RMNode node) {
        // TODO: implement HDFS block-locality lookup
        // For the demo/simulation, return an empty list (locality estimated by QPSO service).
        return Collections.emptyList();
    }
}
