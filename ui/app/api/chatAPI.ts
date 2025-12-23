import { SendMessageParams } from '../store/types'

// Define the base URL for our chat API
// Default to localhost for development, can be overridden via environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const TASKS_ENDPOINT = `${API_BASE_URL}/tasks`
const API_KEY = import.meta.env.VITE_API_KEY || '1234567890'

/**
 * Interface for task creation request
 */
export interface CreateTaskRequest {
  query: string
  conversation_id?: string
  user_id?: string
  agent_plan?: string[]
  metadata?: Record<string, any>
}

/**
 * Interface for task creation response
 */
export interface CreateTaskResponse {
  task_id: string
  status: string
  message: string
}

/**
 * Interface for task state response
 */
export interface TaskStateResponse {
  task_id: string
  status: string
  progress?: {
    workflow_id: string
    status: string
    total_agents: number
    completed_agents: number
    running_agents: number
    failed_agents: number
    agent_statuses: Array<{
      agent_name: string
      agent_category: string
      status: string
      started_at: string
      completed_at?: string
      error?: string
      metadata?: Record<string, any>
    }>
    progress_percentage?: number
    current_step?: string
    iteration_number?: number
    metadata?: Record<string, any>
  }
  state?: {
    workflow_id: string
    status: string
    context: Record<string, any>
    agent_responses: Array<{
      content: string | {
        summary: string
        key_findings?: string[]
        evidence?: Record<string, any>
        confidence?: number
      }
      status: string
      agent_name: string
      agent_category: string
      tool_calls?: Array<Record<string, any>>
      metadata?: Record<string, any>
      error?: string
      timestamp: string
    }>
    shared_data: Record<string, any>
    execution_history: Array<Record<string, any>>
    metadata?: Record<string, any> & {
      iterations_completed?: number
      final_satisfactory?: boolean
    }
  }
}

/**
 * Interface for SSE event data
 */
export interface TaskEvent {
  type: 'status' | 'agent_status' | 'progress' | 'complete' | 'error'
  status?: string
  agent?: string
  progress_percentage?: number
  completed_agents?: number
  total_agents?: number
  iteration_number?: number
  task_id: string
  error?: string
}

/**
 * Creates a new task by posting to the /tasks endpoint
 */
export const createTask = async (request: CreateTaskRequest): Promise<CreateTaskResponse> => {
  const response = await fetch(TASKS_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify(request)
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Failed to create task: ${response.status} - ${errorText}`)
  }

  return response.json()
}

/**
 * Gets the current state of a task
 */
export const getTaskState = async (
  taskId: string,
  includeProgress: boolean = true,
  includeState: boolean = true
): Promise<TaskStateResponse> => {
  const params = new URLSearchParams()
  if (includeProgress) params.append('include_progress', 'true')
  if (includeState) params.append('include_state', 'true')

  const url = `${TASKS_ENDPOINT}/${taskId}${params.toString() ? `?${params.toString()}` : ''}`
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'X-API-Key': API_KEY
    }
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Failed to get task state: ${response.status} - ${errorText}`)
  }

  return response.json()
}

/**
 * Streams task events via Server-Sent Events (SSE)
 * @param taskId The task ID to stream events for
 * @param onEvent Callback function called when an event is received
 * @param abortController Optional AbortController for cancelling the stream
 */
export async function* streamTaskEvents(
  taskId: string,
  onEvent?: (event: TaskEvent) => void,
  abortController?: AbortController
): AsyncGenerator<TaskEvent> {
  const url = `${TASKS_ENDPOINT}/${taskId}/events`
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'X-API-Key': API_KEY
    },
    signal: abortController?.signal
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Failed to stream task events: ${response.status} - ${errorText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Response body is not readable')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        // Process any remaining buffer
        if (buffer.trim()) {
          const events = parseSSEBuffer(buffer)
          for (const event of events) {
            if (onEvent) onEvent(event)
            yield event
          }
        }
        break
      }

      const chunk = decoder.decode(value, { stream: true })
      buffer += chunk

      // Process complete SSE events (separated by \n\n)
      let eventEndIndex
      while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
        const eventData = buffer.substring(0, eventEndIndex)
        buffer = buffer.substring(eventEndIndex + 2)

        const events = parseSSEBuffer(eventData)
        for (const event of events) {
          if (onEvent) onEvent(event)
          yield event
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Parses SSE buffer into TaskEvent objects
 */
function parseSSEBuffer(buffer: string): TaskEvent[] {
  const events: TaskEvent[] = []
  const lines = buffer.split('\n')

  let currentData = ''
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      currentData = line.substring(6).trim()
    } else if (line.trim() === '' && currentData) {
      // Empty line indicates end of event
      try {
        const parsed = JSON.parse(currentData)
        events.push(parsed as TaskEvent)
        currentData = ''
      } catch (e) {
        console.error('Failed to parse SSE event data:', e, 'Data:', currentData)
      }
    }
  }

  // Handle case where buffer ends without trailing newline
  if (currentData) {
    try {
      const parsed = JSON.parse(currentData)
      events.push(parsed as TaskEvent)
    } catch (e) {
      console.error('Failed to parse SSE event data:', e, 'Data:', currentData)
    }
  }

  return events
}

/**
 * Extracts the final response content from a task state
 */
export function extractFinalResponse(taskState: TaskStateResponse): {
  content: string
  sources?: string[]
  metadata?: Record<string, any>
} {
  if (!taskState.state) {
    return { content: 'Task completed but no state available' }
  }

  // Get the final response from agent_responses
  // The workflow returns final_response, but we need to extract it from state
  const agentResponses = taskState.state.agent_responses || []
  
  if (agentResponses.length === 0) {
    return { content: 'Task completed but no agent responses available' }
  }

  // Get the last response (or look for a specific final response)
  const finalResponse = agentResponses[agentResponses.length - 1]
  
  // Extract content - it can be a string or an AgentInsight object
  let content = ''
  if (typeof finalResponse.content === 'string') {
    content = finalResponse.content
  } else if (finalResponse.content && typeof finalResponse.content === 'object') {
    // It's an AgentInsight object
    const insight = finalResponse.content as {
      summary: string
      key_findings?: string[]
      evidence?: Record<string, any>
      confidence?: number
    }
    content = insight.summary || ''
    
    // Key findings are no longer displayed in the chat bubble
  }

  // Extract sources from metadata or tool_calls
  const sources: string[] = []
  if (finalResponse.metadata?.sources) {
    sources.push(...(Array.isArray(finalResponse.metadata.sources) ? finalResponse.metadata.sources : []))
  }
  
  // Extract from tool_calls if available
  if (finalResponse.tool_calls) {
    for (const toolCall of finalResponse.tool_calls) {
      if (toolCall.result?.sources) {
        const toolSources = Array.isArray(toolCall.result.sources) 
          ? toolCall.result.sources 
          : [toolCall.result.sources]
        sources.push(...toolSources)
      }
    }
  }

  return {
    content: content || 'No content available',
    sources: sources.length > 0 ? sources : undefined,
    metadata: finalResponse.metadata
  }
}

// Legacy functions for backward compatibility (deprecated)
/**
 * @deprecated Use createTask and streamTaskEvents instead
 */
export const sendMessage = async ({
  message,
  settings,
  abortController
}: SendMessageParams): Promise<Response> => {
  // This is a legacy function - we'll redirect to the new task-based API
  // Note: execution_mode is not passed - let the orchestration agent decide
  const taskRequest: CreateTaskRequest = {
    query: message,
    metadata: {
      model: settings.model,
      num_results: settings.numRecords || 50,
      generate_alternative_viewpoint: settings.generateAlternativeOpinions
    }
  }

  const taskResponse = await createTask(taskRequest)
  
  // Return a mock Response that can be used with streamTaskEvents
  // This maintains backward compatibility
  return new Response(JSON.stringify(taskResponse), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })
}

/**
 * @deprecated Use streamTaskEvents instead
 */
export async function* processStreamResponse(
  response: Response
): AsyncGenerator<string> {
  // This is a legacy function - should not be used with new API
  throw new Error('processStreamResponse is deprecated. Use streamTaskEvents instead.')
}
