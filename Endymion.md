# Endymion

>   Draft of Hyperion

## Structure

```mermaid

graph TD;
    S[Story Text]-->|info Extractor|I[Styled Infomation]
    I-->|script Template|GS{Game Script} 
    I-->|map Tools|M[Maps]
```

```mermaid

graph TD;
    CJD[Card Json Data]-->|card template Formatter|CP[Card for Print]
    CJD-->|game data Parser|GD{Game Data} 
```