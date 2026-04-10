package org.qips.model;
import java.util.*;

/** Result returned by the QPSO REST service after optimization. */
public class ScheduleResult {
    private final List<String>        orderedJobIds;
    private final double              fitness;
    private final Map<String, Object> metrics;

    public ScheduleResult(List<String> orderedJobIds, double fitness, Map<String, Object> metrics) {
        this.orderedJobIds = Collections.unmodifiableList(orderedJobIds);
        this.fitness       = fitness;
        this.metrics       = Collections.unmodifiableMap(metrics);
    }

    public List<String>        getOrderedJobIds() { return orderedJobIds; }
    public double              getFitness()        { return fitness; }
    public Map<String, Object> getMetrics()        { return metrics; }
}
