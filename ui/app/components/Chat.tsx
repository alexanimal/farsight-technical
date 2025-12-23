import React, { useRef, useEffect } from 'react';
import { MessageList } from './MessageList';
import { InputArea } from './InputArea';
import LoadingIndicator from './LoadingIndicator';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { useChatWithMetadata } from '../hooks/useChatWithMetadata';

/**
 * Main Chat component that combines all chat-related components
 * Includes header, message list, and input area
 */
const Chat: React.FC = () => {
  const { isLoading, clearMessages, conversationId, setConversationId, generateNewConversationId } = useChatStore();
  const settings = useSettingsStore();
  const { messages, streamingMetadata } = useChatWithMetadata();
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // Extract iteration info from streaming metadata
  const currentIteration = streamingMetadata?.iteration_number;
  const iterationsCompleted = streamingMetadata?.iterations_completed;

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      // Slight delay to ensure content is rendered
      setTimeout(() => {
        if (chatContainerRef.current) {
          chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
      }, 100);
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Header */}
      <header className="border-b dark:border-gray-700 p-4 bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {isLoading 
                ? (currentIteration 
                    ? `Processing... (Iteration ${currentIteration}/3)`
                    : 'Processing...')
                : 'Ready'}
            </span>
            {isLoading && <LoadingIndicator size="small" />}

            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to start a new conversation? This will clear all messages.')) {
                  clearMessages();
                  generateNewConversationId();
                }
              }}
              className="text-sm px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded-md transition-colors"
              aria-label="Start new conversation"
            >
              New Conversation
            </button>

            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all messages? This cannot be undone.')) {
                  clearMessages();
                }
              }}
              className="text-sm px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded-md transition-colors"
              aria-label="Clear chat history"
            >
              Clear Chat
            </button>
          </div>
        </div>
      </header>

      {/* Message List - Flex grow to take available space */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto relative"
      >
        <MessageList />
      </div>

      {/* Input Area - Fixed at bottom */}
      <InputArea />
    </div>
  );
};

export default Chat;
