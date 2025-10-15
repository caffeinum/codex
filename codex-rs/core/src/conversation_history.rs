use std::collections::HashMap;
use std::collections::HashSet;

use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use tracing::warn;

use crate::state::TaskKind;

const MAIN_THREAD_KEY: &str = "main";

/// Transcript of conversation history
#[derive(Debug, Clone)]
pub(crate) struct ConversationHistory {
    /// Each entry stores the oldest item at index 0.
    threads: HashMap<String, Vec<ResponseItem>>,
}

impl ConversationHistory {
    pub(crate) fn new() -> Self {
        Self {
            threads: HashMap::from([(MAIN_THREAD_KEY.to_string(), Vec::new())]),
        }
    }

    /// Returns a clone of the contents in the transcript.
    pub(crate) fn contents(&self) -> Vec<ResponseItem> {
        self.thread_snapshot(MAIN_THREAD_KEY)
    }

    pub(crate) fn clear_task_history(&mut self, task_kind: TaskKind) {
        self.clear_thread(task_kind.history_key());
    }

    /// `items` is ordered from oldest to newest.
    pub(crate) fn record_items<I>(&mut self, items: I, task_kind: TaskKind)
    where
        I: IntoIterator<Item = ResponseItem>,
    {
        self.record_items_for_key(items, task_kind.history_key());
    }

    pub(crate) fn record_items_for_key<I>(&mut self, items: I, key: &str)
    where
        I: IntoIterator<Item = ResponseItem>,
    {
        let thread = self.thread_mut(key);
        for item in items {
            if !is_api_message(&item) {
                continue;
            }

            thread.push(item);
        }
    }

    pub(crate) fn replace(&mut self, items: Vec<ResponseItem>) {
        self.threads.insert(MAIN_THREAD_KEY.to_string(), items);
    }

    pub(crate) fn add_pending_input(
        &mut self,
        pending_input: Vec<ResponseItem>,
        task_kind: TaskKind,
    ) {
        self.record_items(pending_input, task_kind);
    }

    pub(crate) fn initialize_task_history(
        &mut self,
        task_kind: TaskKind,
        response_input: &ResponseInputItem,
        initial_context: Vec<ResponseItem>,
    ) {
        self.initialize_thread(task_kind.history_key(), response_input, initial_context);
    }

    pub(crate) fn handle_missing_tool_call_output(&mut self, task_kind: TaskKind) {
        let key = task_kind.history_key();
        // call_ids that are part of this response.
        let content = self.thread_snapshot(key);
        let completed_call_ids: HashSet<String> = content
            .iter()
            .filter_map(|ri| match ri {
                ResponseItem::FunctionCallOutput { call_id, .. } => Some(call_id.clone()),
                ResponseItem::CustomToolCallOutput { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();

        // call_ids that were pending but are not part of this response.
        // This usually happens because the user interrupted the model before we responded to one of its tool calls
        // and then the user sent a follow-up message.
        let missing_call_ids: Vec<String> = content
            .iter()
            .filter_map(|ri| match ri {
                ResponseItem::FunctionCall { call_id, .. } => Some(call_id),
                ResponseItem::LocalShellCall {
                    call_id: Some(call_id),
                    ..
                } => Some(call_id),
                ResponseItem::CustomToolCall { call_id, .. } => Some(call_id),
                _ => None,
            })
            .filter(|call_id| !completed_call_ids.contains(*call_id))
            .cloned()
            .collect();

        if missing_call_ids.is_empty() {
            return;
        }

        warn!(
            history_key = key,
            missing_call_ids = ?missing_call_ids,
            "detected tool calls without outputs; inserting synthetic aborted outputs"
        );

        let missing_calls = missing_call_ids
            .iter()
            .map(|call_id| ResponseItem::CustomToolCallOutput {
                call_id: call_id.clone(),
                output: "aborted".to_string(),
            })
            .collect::<Vec<_>>();

        self.record_items_for_key(missing_calls, key);
    }

    pub(crate) fn prompt(&self, task_kind: TaskKind) -> Vec<ResponseItem> {
        self.thread_snapshot(task_kind.history_key())
    }

    fn initialize_thread(
        &mut self,
        key: &str,
        response_input: &ResponseInputItem,
        initial_context: Vec<ResponseItem>,
    ) {
        self.clear_thread(key);
        self.record_items_for_key(initial_context, key);
        self.record_items_for_key(
            std::iter::once(ResponseItem::from(response_input.clone())),
            key,
        );
    }

    fn thread_snapshot(&self, key: &str) -> Vec<ResponseItem> {
        self.threads.get(key).cloned().unwrap_or_else(Vec::new)
    }

    fn thread_mut(&mut self, key: &str) -> &mut Vec<ResponseItem> {
        self.threads.entry(key.to_string()).or_default()
    }

    fn clear_thread(&mut self, key: &str) {
        if let Some(thread) = self.threads.get_mut(key) {
            thread.clear();
        }
    }
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Anything that is not a system message or "reasoning" message is considered
/// an API message.
fn is_api_message(message: &ResponseItem) -> bool {
    match message {
        ResponseItem::Message { role, .. } => role.as_str() != "system",
        ResponseItem::FunctionCallOutput { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::CustomToolCall { .. }
        | ResponseItem::CustomToolCallOutput { .. }
        | ResponseItem::LocalShellCall { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::WebSearchCall { .. } => true,
        ResponseItem::Other => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use pretty_assertions::assert_eq;

    fn assistant_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
        }
    }

    fn user_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
        }
    }

    #[test]
    fn filters_non_api_messages() {
        let mut h = ConversationHistory::new();
        // System message is not an API message; Other is ignored.
        let system = ResponseItem::Message {
            id: None,
            role: "system".to_string(),
            content: vec![ContentItem::OutputText {
                text: "ignored".to_string(),
            }],
        };
        h.record_items([system, ResponseItem::Other], TaskKind::Regular);

        // User and assistant should be retained.
        let u = user_msg("hi");
        let a = assistant_msg("hello");
        h.record_items([u, a], TaskKind::Regular);

        let items = h.contents();
        assert_eq!(
            items,
            vec![
                ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: "hi".to_string()
                    }]
                },
                ResponseItem::Message {
                    id: None,
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: "hello".to_string()
                    }]
                }
            ],
        );
    }

    #[test]
    fn inserts_missing_tool_call_output_once() {
        let mut h = ConversationHistory::new();
        let call_id = "call-1".to_string();
        let tool_call = ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: call_id.clone(),
            name: "example".to_string(),
            input: "{}".to_string(),
        };
        h.record_items([tool_call.clone()], TaskKind::Regular);
        h.handle_missing_tool_call_output(TaskKind::Regular);

        let expected = vec![
            tool_call,
            ResponseItem::CustomToolCallOutput {
                call_id,
                output: "aborted".to_string(),
            },
        ];
        assert_eq!(h.contents(), expected);

        // A second pass should be a no-op because the synthetic output now exists.
        h.handle_missing_tool_call_output(TaskKind::Regular);
        assert_eq!(h.contents(), expected);
    }
}
