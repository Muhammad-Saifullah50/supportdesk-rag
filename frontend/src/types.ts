export interface Citation {
  id: string;
  document_id: string;
  label: string;
  pdf_filename: string;
  pdf_url: string;
  page: number;
  section: string;
  highlight_text: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
