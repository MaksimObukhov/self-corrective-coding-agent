from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = 'local'
    langchain_tracing_v2: bool | None = False
    langchain_api_key: SecretStr | None = None
    langchain_tracing_tags: list[str] | None = Field(default_factory=list)


CONFIG = Settings()
