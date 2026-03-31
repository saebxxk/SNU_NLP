# table_parsers/__init__.py

from .base import (
    BaseTableParser,
    StatementRelatedTableParser,
    StatementBodyTableParser,
    TextLikeTableParser,
    MatrixLikeTableParser,
)

from .statement import (
    StatementTitleTableParser,
    StatementBodyTypeAParser,
    ChangesInEquityParser,
    StatementFooterTableParser,
)

from .note_tables import (
    RollForwardTableParser,
    FinancialInstrumentTableParser,
    CounterpartyTableParser,
    DividendTableParser,
    PensionTableParser,
    InventoryValuationTableParser,
    ExpenseBreakdownTableParser,
    EnvironmentalTableParser,
)

from .misc import (
    MetadataTableParser,
    NoteTableParser,
    AuditRelatedTableParser,
    SimpleMatrixTableParser,
    UnknownTableParser,
)