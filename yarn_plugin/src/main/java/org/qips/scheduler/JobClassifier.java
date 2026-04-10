package org.qips.scheduler;

import org.qips.model.JobMetadata;
import java.util.*;

/**
 * JobClassifier — assigns normalized priority scores and deadline tags
 * to incoming YARN jobs before they enter the QPSO optimizer.
 *
 * Priority is derived from:
 *   1. YARN application priority (0–10 scale)
 *   2. Queue name (e.g. "high-priority" queues get a boost)
 *   3. Job type tags set by the submitter
 *   4. Time-in-queue (aging: jobs waiting longer get a gradual boost)
 *
 * Deadline slack is extracted from an application tag in the format:
 *   deadline=<unix_epoch_ms>
 * If absent, a default slack of 1 hour is assumed.
 */
public class JobClassifier {

    private static final double DEFAULT_DEADLINE_SLACK_SEC = 3600.0;
    private static final double AGING_BOOST_PER_MINUTE     = 0.01;  // +1% priority per min waiting
    private static final double MAX_AGING_BOOST            = 0.30;  // cap at +30%

    /** Queue names that receive a priority multiplier. */
    private static final Map<String, Double> QUEUE_MULTIPLIERS = Map.of(
        "critical",      1.0,
        "high",          0.75,
        "default",       0.4,
        "low",           0.15,
        "batch",         0.1
    );

    /**
     * Enrich a list of raw job metadata with normalized priorities and deadlines.
     *
     * @param jobs  raw job metadata from TaskProfiler
     * @return      same list, enriched in place
     */
    public List<JobMetadata> classify(List<JobMetadata> jobs) {
        long now = System.currentTimeMillis();
        for (JobMetadata job : jobs) {
            double priority = computePriority(job, now);
            double deadline = extractDeadlineSlack(job, now);
            job.setNormalizedPriority(priority);
            job.setDeadlineSlack(deadline);
        }
        // Sort descending by priority so the QPSO warm-starts with a good initial guess
        jobs.sort(Comparator.comparingDouble(JobMetadata::getNormalizedPriority).reversed());
        return jobs;
    }

    /**
     * Compute a normalized priority in [0, 1] for a job.
     *
     * Formula:
     *   base     = yarn_priority / 10.0
     *   queue    = QUEUE_MULTIPLIERS.getOrDefault(queueName, 0.4)
     *   aging    = min(MAX_AGING_BOOST, wait_minutes * AGING_BOOST_PER_MINUTE)
     *   priority = clamp(base * 0.5 + queue * 0.4 + aging * 0.1, 0, 1)
     */
    private double computePriority(JobMetadata job, long nowMs) {
        double base       = Math.min(1.0, job.getYarnPriority() / 10.0);
        double queueBoost = QUEUE_MULTIPLIERS.getOrDefault(
            job.getQueueName().toLowerCase(), 0.4);
        double waitMin    = (nowMs - job.getSubmitTimeMs()) / 60_000.0;
        double aging      = Math.min(MAX_AGING_BOOST, waitMin * AGING_BOOST_PER_MINUTE);

        double raw = base * 0.5 + queueBoost * 0.4 + aging * 0.1;
        return Math.max(0.0, Math.min(1.0, raw));
    }

    /**
     * Extract deadline slack (seconds from now until deadline).
     * Looks for an application tag in the format: deadline=<epoch_ms>
     * Falls back to DEFAULT_DEADLINE_SLACK_SEC if not found.
     */
    private double extractDeadlineSlack(JobMetadata job, long nowMs) {
        for (String tag : job.getApplicationTags()) {
            if (tag.startsWith("deadline=")) {
                try {
                    long deadlineMs = Long.parseLong(tag.substring(9));
                    double slackSec = (deadlineMs - nowMs) / 1000.0;
                    return Math.max(1.0, slackSec);
                } catch (NumberFormatException ignored) {}
            }
        }
        return DEFAULT_DEADLINE_SLACK_SEC;
    }
}
