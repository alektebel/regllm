import { ChatWindow } from "@/components/chat/ChatWindow";

interface Props {
  params: { id?: string[] };
}

export default function ChatPage({ params }: Props) {
  const conversationId = params.id?.[0] ? parseInt(params.id[0]) : undefined;
  return <ChatWindow conversationId={conversationId} />;
}
