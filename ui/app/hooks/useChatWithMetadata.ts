import { useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { createTask, streamTaskEvents, getTaskState, extractFinalResponse, TaskEvent } from '../api/chatAPI';
import { useStreamingMessage } from './useStreamingMessage';
import { ChatMessage } from '../store/types';

/**
 * Enhanced chat hook that properly handles metadata in both streaming and non-streaming responses
 * Updated to work with the new task-based API
 */
export const useChatWithMetadata = () => {
  const {
    messages,
    addMessage,
    setLoading,
    isLoading,
  } = useChatStore();

  // Get streaming utilities from our custom hook
  const {
    updateStreamingContent,
    setStreamingMetadata,
    setStreamingSources,
    setStreamingAlternativeViewpoint,
    handleStreamComplete,
    finalizeStreamingMessageWithMetadata,
    resetStreaming,
    pendingMetadata,
    pendingSources,
    pendingAlternativeViewpoint,
    isStreamComplete
  } = useStreamingMessage();

  const settings = useSettingsStore();
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentTaskIdRef = useRef<string | null>(null);

  // Debug effect to monitor changes in streaming metadata
  useEffect(() => {
    if (isStreamComplete) {
      console.log('Stream completed status changed with metadata:', {
        hasSources: Array.isArray(pendingSources) && pendingSources.length > 0,
        sourcesCount: pendingSources.length,
        hasAlternativeViewpoint: !!pendingAlternativeViewpoint,
        metadataAvailable: !!pendingMetadata
      });
    }
  }, [isStreamComplete, pendingSources, pendingAlternativeViewpoint, pendingMetadata]);

  /**
   * Sends a user message to the chat API
   * Creates a task and streams events, then fetches final state when complete
   */
  const sendUserMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    // Create user message
    const userMessage: ChatMessage = {
      role: 'user',
      content,
      id: uuidv4(),
      timestamp: Date.now()
    };

    addMessage(userMessage);
    setLoading(true);
    resetStreaming();

    try {
      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController();

      console.log('Creating task for query:', content);

      // Step 1: Create the task
      // Note: execution_mode is not passed - let the orchestration agent decide
      const taskResponse = await createTask({
        query: content,
        metadata: {
          model: settings.model,
          num_results: settings.numRecords || 50,
          generate_alternative_viewpoint: settings.generateAlternativeOpinions
        }
      });

      currentTaskIdRef.current = taskResponse.task_id;
      console.log('Task created:', taskResponse.task_id);

      // Step 2: Stream events from the task
      if (settings.apiType === 'stream') {
        console.log('Streaming task events');
        
        let lastProgress = 0;
        let statusMessage = 'Processing your request...';

        try {
          // Update UI with initial status
          updateStreamingContent(statusMessage);

          // Stream events
          for await (const event of streamTaskEvents(
            taskResponse.task_id,
            (event) => {
              console.log('Received task event:', event);
            },
            abortControllerRef.current
          )) {
            // Handle different event types
            switch (event.type) {
              case 'status':
                statusMessage = `Status: ${event.status || 'processing'}...`;
                updateStreamingContent(statusMessage);
                break;

              case 'agent_status':
                statusMessage = `${event.agent || 'Agent'} is ${event.status || 'working'}...`;
                updateStreamingContent(statusMessage);
                break;

              case 'progress':
                if (event.progress_percentage !== undefined) {
                  lastProgress = event.progress_percentage;
                  statusMessage = `Processing... ${event.progress_percentage}% (${event.completed_agents || 0}/${event.total_agents || 0} agents completed)`;
                  updateStreamingContent(statusMessage);
                }
                break;

              case 'complete':
                console.log('Task completed, fetching final state...');
                // Task is complete, fetch the final state
                statusMessage = 'Finalizing response...';
                updateStreamingContent(statusMessage);
                break;

              case 'error':
                throw new Error(event.error || 'Task execution failed');
            }

            // If task is complete, break out of the loop
            if (event.type === 'complete') {
              break;
            }
          }

          // Step 3: Fetch the final task state to get the response
          console.log('Fetching final task state...');
          const finalState = await getTaskState(taskResponse.task_id, true, true);
          
          console.log('Final task state received:', {
            status: finalState.status,
            hasState: !!finalState.state,
            agentResponsesCount: finalState.state?.agent_responses?.length || 0
          });

          // Extract the final response content
          const finalResponse = extractFinalResponse(finalState);
          
          console.log('Extracted final response:', {
            contentLength: finalResponse.content.length,
            hasSources: Array.isArray(finalResponse.sources) && finalResponse.sources.length > 0,
            sourcesCount: finalResponse.sources?.length || 0
          });

          // Update streaming content with the final response
          updateStreamingContent(finalResponse.content);

          // Store sources if available
          if (finalResponse.sources && finalResponse.sources.length > 0) {
            setStreamingSources(finalResponse.sources);
          }

          // Store metadata
          if (finalResponse.metadata) {
            setStreamingMetadata(finalResponse.metadata);
          }

          // Finalize the message
          handleStreamComplete({
            content: finalResponse.content,
            isComplete: true,
            sources: finalResponse.sources,
            metadata: finalResponse.metadata
          });

        } catch (streamError: any) {
          if (streamError.name === 'AbortError') {
            console.log('Stream cancelled by user');
            return;
          }
          
          console.error('Error processing task stream:', streamError);
          resetStreaming();
          addMessage({
            role: 'system',
            content: `Error processing task: ${streamError.message || 'Unknown error'}`,
            id: uuidv4(),
            timestamp: Date.now()
          });
        }
      } else {
        // Non-streaming mode: wait for completion and fetch state
        console.log('Waiting for task completion (non-streaming mode)...');
        
        // Poll for completion
        let attempts = 0;
        const maxAttempts = 60; // 60 seconds max wait
        let isComplete = false;

        while (!isComplete && attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
          
          try {
            const taskState = await getTaskState(taskResponse.task_id, false, false);
            
            if (taskState.status === 'completed' || taskState.status === 'failed' || taskState.status === 'cancelled') {
              isComplete = true;
              
              // Fetch full state
              const finalState = await getTaskState(taskResponse.task_id, true, true);
              const finalResponse = extractFinalResponse(finalState);
              
              addMessage({
                role: 'assistant',
                content: finalResponse.content,
                id: uuidv4(),
                timestamp: Date.now(),
                sources: finalResponse.sources,
                alternativeViewpoint: undefined
              });
            }
          } catch (error) {
            console.error('Error polling task state:', error);
            attempts++;
          }
          
          attempts++;
        }

        if (!isComplete) {
          throw new Error('Task did not complete within timeout period');
        }
      }
    } catch (error: any) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        console.error('Error sending message:', error);
        addMessage({
          role: 'system',
          content: `Error: ${error.message || 'Failed to send message. Please try again.'}`,
          id: uuidv4(),
          timestamp: Date.now()
        });
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
      currentTaskIdRef.current = null;
    }
  }, [
    addMessage,
    setLoading,
    settings,
    updateStreamingContent,
    handleStreamComplete,
    finalizeStreamingMessageWithMetadata,
    resetStreaming,
    setStreamingSources,
    setStreamingMetadata,
    isStreamComplete
  ]);

  /**
   * Cancels the current request if one is in progress
   */
  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      resetStreaming();
    }
  }, [setLoading, resetStreaming]);

  return {
    messages,
    sendMessage: sendUserMessage,
    cancelRequest,
    isLoading,
    streamingMetadata: pendingMetadata,
    streamingSources: pendingSources,
    streamingAlternativeViewpoint: pendingAlternativeViewpoint,
    isStreamComplete
  };
};

export default useChatWithMetadata;
