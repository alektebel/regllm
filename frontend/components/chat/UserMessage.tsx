interface Props {
  content: string;
}

export function UserMessage({ content }: Props) {
  return (
    <div className="flex gap-3 py-4 justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary/15 px-4 py-2.5 text-sm text-foreground whitespace-pre-wrap">
        {content}
      </div>
      <div className="shrink-0 w-7 h-7 rounded-full bg-secondary flex items-center justify-center text-xs font-bold mt-0.5">
        U
      </div>
    </div>
  );
}
