flowchart TD
    START([User Topic]) --> R[Researcher Node<br/>Generates queries & fetches web snippets]
    R --> W[Writer Node<br/>Drafts cited technical brief from sources only]
    W --> C[Critic Node<br/>Audits claims vs. raw snippets via Pydantic JSON]
    C -->|passed == False AND revisions left| W
    C -->|passed == True OR max_revisions hit| END([Final Verified Report + Audit Log])

