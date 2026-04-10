package org.qips.model;

import java.util.List;
import java.util.ArrayList;

public class JobMetadata {
    private String id, name, queueName;
    private int yarnPriority, cpuDemand;
    private long memDemand, submitTimeMs;
    private double estimatedRuntime, inputSizeGb, normalizedPriority, deadlineSlack;
    private List<String> applicationTags;

    public JobMetadata(String id, String name, int yarnPriority, String queueName,
                       long submitTimeMs, double estimatedRuntime, int cpuDemand,
                       long memDemand, double deadlineSlack, double normalizedPriority,
                       double inputSizeGb, List<String> applicationTags) {
        this.id=id; this.name=name; this.yarnPriority=yarnPriority;
        this.queueName=queueName!=null?queueName:"default";
        this.submitTimeMs=submitTimeMs; this.estimatedRuntime=estimatedRuntime;
        this.cpuDemand=cpuDemand; this.memDemand=memDemand;
        this.deadlineSlack=deadlineSlack; this.normalizedPriority=normalizedPriority;
        this.inputSizeGb=inputSizeGb;
        this.applicationTags=applicationTags!=null?applicationTags:new ArrayList<>();
    }

    public String getId()                 { return id; }
    public String getName()               { return name; }
    public int    getYarnPriority()       { return yarnPriority; }
    public String getQueueName()          { return queueName; }
    public long   getSubmitTimeMs()       { return submitTimeMs; }
    public double getEstimatedRuntime()   { return estimatedRuntime; }
    public int    getCpuDemand()          { return cpuDemand; }
    public long   getMemDemand()          { return memDemand; }
    public double getDeadlineSlack()      { return deadlineSlack; }
    public double getNormalizedPriority() { return normalizedPriority; }
    public double getInputSizeGb()        { return inputSizeGb; }
    public List<String> getApplicationTags() { return applicationTags; }
    public double getPriority()           { return normalizedPriority; }
    public void setNormalizedPriority(double v) { this.normalizedPriority = v; }
    public void setDeadlineSlack(double v)      { this.deadlineSlack = v; }
    public void setEstimatedRuntime(double v)   { this.estimatedRuntime = v; }
}
