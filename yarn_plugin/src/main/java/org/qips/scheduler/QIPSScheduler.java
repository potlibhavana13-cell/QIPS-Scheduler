package org.qips.scheduler;

import org.apache.hadoop.yarn.server.resourcemanager.scheduler.AbstractYarnScheduler;
import org.apache.hadoop.yarn.server.resourcemanager.rmnode.RMNode;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.RMApp;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.RMAppAttempt;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.conf.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.qips.rest.QpsoRestClient;
import org.qips.model.JobMetadata;
import org.qips.model.ScheduleResult;

import java.util.*;

/**
 * QIPSScheduler — Quantum-Inspired Priority Scheduler for Hadoop YARN.
 *
 * Integrates with YARN's ResourceManager via a custom scheduler plugin.
 * On each NodeManager heartbeat, collects pending job metadata, calls
 * the external Python QPSO service for optimization, then assigns
 * containers in the returned order.
 *
 * Configure in yarn-site.xml:
 *   yarn.resourcemanager.scheduler.class = org.qips.scheduler.QIPSScheduler
 *   qips.optimizer.url                   = http://localhost:8080/optimize
 */
public class QIPSScheduler extends AbstractYarnScheduler<QIPSAppAttempt, QIPSNode> {

    private static final Logger LOG = LoggerFactory.getLogger(QIPSScheduler.class);

    private QpsoRestClient qpsoClient;
    private JobClassifier  jobClassifier;
    private TaskProfiler   taskProfiler;

    // Fitness weights — tunable via yarn-site.xml
    private double[] weights = {1.0, 1.0, 1.5, 1.0, 1.2, 1.3};

    @Override
    public void serviceInit(Configuration conf) throws Exception {
        super.serviceInit(conf);
        String qpsoUrl = conf.get("qips.optimizer.url", "http://localhost:8080/optimize");
        this.qpsoClient    = new QpsoRestClient(qpsoUrl);
        this.jobClassifier = new JobClassifier();
        this.taskProfiler  = new TaskProfiler(rmContext);

        // Load fitness weights from config if provided
        String wStr = conf.get("qips.fitness.weights", "");
        if (!wStr.isEmpty()) {
            String[] parts = wStr.split(",");
            for (int i = 0; i < Math.min(parts.length, 6); i++) {
                weights[i] = Double.parseDouble(parts[i].trim());
            }
        }
        LOG.info("QIPS Scheduler initialized. QPSO endpoint: {}", qpsoUrl);
    }

    /**
     * Called by YARN on every NodeManager heartbeat.
     * This is the main scheduling entry point.
     */
    @Override
    public synchronized void nodeUpdate(RMNode rmNode) {
        super.nodeUpdate(rmNode);
        try {
            scheduleContainers(rmNode);
        } catch (Exception e) {
            LOG.error("QPSO scheduling failed on node {}, falling back to FIFO",
                       rmNode.getNodeID(), e);
            fifoFallback(rmNode);
        }
    }

    private void scheduleContainers(RMNode rmNode) throws Exception {
        // 1. Collect pending jobs
        List<JobMetadata> pendingJobs = taskProfiler.collectPendingJobs();
        if (pendingJobs.isEmpty()) return;

        // 2. Classify priorities and deadlines
        pendingJobs = jobClassifier.classify(pendingJobs);

        // 3. Build node metadata
        List<Map<String, Object>> nodeInfo = taskProfiler.collectNodeInfo();

        // 4. Call QPSO optimizer
        LOG.debug("Calling QPSO optimizer with {} pending jobs", pendingJobs.size());
        ScheduleResult result = qpsoClient.optimize(pendingJobs, nodeInfo, weights);

        // 5. Assign containers in optimized order
        assignContainersInOrder(result.getOrderedJobIds(), rmNode);

        LOG.info("QIPS: scheduled {} jobs on {}. Fitness: {}",
                  result.getOrderedJobIds().size(), rmNode.getNodeID(), result.getFitness());
    }

    /**
     * Assign containers to NodeManagers in the QPSO-optimized job order.
     * Respects resource constraints — skips jobs that don't fit current capacity.
     */
    private void assignContainersInOrder(List<String> orderedJobIds, RMNode node) {
        for (String jobId : orderedJobIds) {
            try {
                ApplicationId appId = ApplicationId.fromString(jobId);
                QIPSAppAttempt attempt = getApplicationAttempt(appId);
                if (attempt == null || !attempt.hasPendingResourceRequests()) continue;

                // Check node has capacity
                if (nodeHasCapacity(node, attempt.getNextResourceRequest())) {
                    allocateContainer(attempt, node);
                    LOG.debug("Allocated container for {} on {}", jobId, node.getNodeID());
                }
            } catch (Exception e) {
                LOG.warn("Failed to allocate container for job {}: {}", jobId, e.getMessage());
            }
        }
    }

    /** Simple FIFO fallback when QPSO service is unreachable. */
    private void fifoFallback(RMNode rmNode) {
        List<JobMetadata> pending = taskProfiler.collectPendingJobs();
        for (JobMetadata job : pending) {
            try {
                ApplicationId appId = ApplicationId.fromString(job.getId());
                QIPSAppAttempt attempt = getApplicationAttempt(appId);
                if (attempt != null && attempt.hasPendingResourceRequests()
                        && nodeHasCapacity(rmNode, attempt.getNextResourceRequest())) {
                    allocateContainer(attempt, rmNode);
                }
            } catch (Exception ignored) {}
        }
    }

    // ── Stubs — fill in for real Hadoop 3.3.4 integration ────────────────────

    private QIPSAppAttempt getApplicationAttempt(ApplicationId appId) {
        return null; // TODO: return from scheduler's app map
    }

    private boolean nodeHasCapacity(RMNode node, Object request) {
        return true; // TODO: compare Resources
    }

    private void allocateContainer(QIPSAppAttempt attempt, RMNode node) {
        // TODO: call ApplicationMasterService to launch container
    }
}
