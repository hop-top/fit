package fit

import "hop.top/kit/bus"

// Event topics for trace lifecycle.
const (
	TopicTraceCreated   bus.Topic = "fit.trace.created"
	TopicTraceBatch     bus.Topic = "fit.trace.batch"
	TopicAdvisorUpdated bus.Topic = "fit.advisor.updated"
)

func newTraceEvent(source string, trace *Trace) bus.Event {
	return bus.NewEvent(TopicTraceCreated, source, trace)
}
