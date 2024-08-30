PROMPT_SV = [
    """Nedan finns ett utdrag från en webbsida på {language}. Utvärdera huruvida sidan har ett högt pedagogiskt värde och skulle kunna vara användbar i en utbildningsmiljö för undervisning med hjälp av det additiva 4-poängssystemet som beskrivs nedan. Poäng ackumuleras baserat på uppfyllandet av varje kriterium:

- Lägg till 1 poäng om utdraget innehåller viss omvärldsinformation som är relevant för utbildningsämnen, även om det inkluderar en mindre mängd irrelevant eller icke-akademiskt innehåll som annonser och marknadsföringsmaterial.
- Lägg till ytterligare en poäng om utdraget är lämpligt för utbildningsändamål och introducerar nyckelbegrepp som är relevanta inom skolplaner. Texten är sammanhängande även om innehållet inte nödändigtvis är heltäckande. Den kan innehålla viss överflödig information. 
- Tilldela en tredje poäng om utdraget är högst relevant för utbildningsändamål och skriven på ett på ett tydligt och sammanhängande sätt. Det kan likna ett kapitel i en lärobok eller en handledning och innehåller betydande pedagogiskt innehåll, som exempelvis övningar och lösningar, med minimalt irrelevant innehåll.
- Ge en fjärde poäng om utdraget är utmärkt i sitt pedagogiska värde och passar perfekt för undervisning. Det följer detaljerade resonemang. Texten är lätt att följa och erbjuder djupgående och genomtänkta insikter i ämnet, utan inslag av icke-pedagogiskt innehåll som använder sig av allt för komplexa begrepp.

Om texten är osammanhängande, till större delen innehåller upplistningar utan informativt innehåll, eller endast innehåller ren SEO- eller marknadsföringsmaterial, ge då 0 poäng.

Utdraget:
""",
    """
Efter att ha undersökt utdraget:

* Motivera först kortfattat din totala poäng, upp till 50 ord.
* Avsluta ditt svar med poängen i formatet: "Pedagogiskt värde: <total poäng>""",
]

PROMPT_FINEWEB_JSON_SV = [
    """Nedan är ett utdrag från en webbsida. Utvärdera om sidan har ett högt pedagogiskt värde och kan vara användbar i en pedagogisk miljö för undervisning från grundskola upp till gymnasieskola med hjälp av det additiva 5-punktssystemet som beskrivs nedan. Poäng ackumuleras baserat på uppfyllandet av varje kriterium:

- Lägg till 1 poäng om utdraget ger viss grundläggande information som är relevant för utbildningsämnen, även om det inkluderar viss irrelevant eller icke-akademiskt innehåll som annonser och reklammaterial.
- Lägg till ytterligare en poäng om utdraget tar upp vissa element som är relevanta för utbildning men som inte stämmer överens med utbildningsstandarder. Det kan blanda pedagogiskt innehåll med icke-pedagogiskt material, erbjuda en ytlig översikt över potentiellt användbara ämnen eller presentera information på ett oorganiserat sätt och med en osammanhängande skrivstil.
- Tilldela en tredje poäng om utdraget är lämpligt för utbildningsändamål och introducerar nyckelkoncept som är relevanta för skolans eller gymnasieskolans läroplaner. Det är sammanhängande även om det kanske inte är omfattande eller kan inkludera viss överflödig information. Det kan likna en inledande del av en lärobok eller en grundläggande handledning som är lämplig för lärande men har märkbara begränsningar som att behandla komplexa koncept ytligt.
- Ge en fjärde poäng om utdraget är mycket relevant och fördelaktigt för pedagogiska ändamål och uppvisar en klar och konsekvent skrivstil. Det kan vara likt ett kapitel från en lärobok eller en detaljerad handledning som erbjuder omfattande pedagogiskt innehåll, inklusive övningar och lösningar, med minimal irrelevant information. Innehållet är sammanhängande, fokuserat och värdefullt för strukturerat lärande.
- Tilldela en femte poäng om utdraget är enastående i sitt pedagogiska värde, perfekt lämpat för undervisning upp till gymnasieskola. Det följer detaljerad logik, skrivstilen är lätt att följa och erbjuder djupgående och grundliga insikter i ämnet, utan icke-pedagogiskt eller överdrivet komplext innehåll.

Utdraget: """,
    """
Efter att ha granskat utdraget:

* Motivera kort din totala poäng, upp till 100 ord.
* Ditt svar ska vara i json med fälten \"reason\" och \"educational_score\", där den första är en sträng och den andra ett heltal.""",
]

PROMPT_FINEWEB_SV = [
    """Nedan är ett utdrag från en webbsida. Utvärdera om sidan har ett högt pedagogiskt värde och kan vara användbar i en pedagogisk miljö för undervisning från grundskola upp till gymnasieskola med hjälp av det additiva 5-punktssystemet som beskrivs nedan. Poäng ackumuleras baserat på uppfyllandet av varje kriterium:

- Lägg till 1 poäng om utdraget ger viss grundläggande information som är relevant för utbildningsämnen, även om det inkluderar viss irrelevant eller icke-akademiskt innehåll som annonser och reklammaterial.
- Lägg till ytterligare en poäng om utdraget tar upp vissa element som är relevanta för utbildning men som inte stämmer överens med utbildningsstandarder. Det kan blanda pedagogiskt innehåll med icke-pedagogiskt material, erbjuda en ytlig översikt över potentiellt användbara ämnen eller presentera information på ett oorganiserat sätt och med en osammanhängande skrivstil.
- Tilldela en tredje poäng om utdraget är lämpligt för utbildningsändamål och introducerar nyckelkoncept som är relevanta för skolans eller gymnasieskolans läroplaner. Det är sammanhängande även om det kanske inte är omfattande eller kan inkludera viss överflödig information. Det kan likna en inledande del av en lärobok eller en grundläggande handledning som är lämplig för lärande men har märkbara begränsningar som att behandla komplexa koncept ytligt.
- Ge en fjärde poäng om utdraget är mycket relevant och fördelaktigt för pedagogiska ändamål och uppvisar en klar och konsekvent skrivstil. Det kan vara likt ett kapitel från en lärobok eller en detaljerad handledning som erbjuder omfattande pedagogiskt innehåll, inklusive övningar och lösningar, med minimal irrelevant information. Innehållet är sammanhängande, fokuserat och värdefullt för strukturerat lärande.
- Tilldela en femte poäng om utdraget är enastående i sitt pedagogiska värde, perfekt lämpat för undervisning upp till gymnasieskola. Det följer detaljerad logik, skrivstilen är lätt att följa och erbjuder djupgående och grundliga insikter i ämnet, utan icke-pedagogiskt eller överdrivet komplext innehåll.

Utdraget: """,
    """
Efter att ha granskat utdraget:

* Motivera först kortfattat din totala poäng, upp till 100 ord.
* Avsluta ditt svar med poängen i formatet: \"Pedagogiskt värde: <total poäng>\"""",
]

PROMPT_FINEWEB_LANG_SV = [
    """Nedan är ett utdrag från en webbsida. Utvärdera om sidan har ett högt pedagogiskt värde och kan vara användbar i en pedagogisk miljö för undervisning från grundskola upp till gymnasieskola med hjälp av det additiva 5-punktssystemet som beskrivs nedan. Poäng ackumuleras baserat på uppfyllandet av varje kriterium:

- Lägg till 1 poäng om utdraget ger viss grundläggande information som är relevant för utbildningsämnen, även om det inkluderar viss irrelevant eller icke-akademiskt innehåll som annonser och reklammaterial.
- Lägg till ytterligare en poäng om utdraget tar upp vissa element som är relevanta för utbildning men som inte stämmer överens med utbildningsstandarder. Det kan blanda pedagogiskt innehåll med icke-pedagogiskt material, erbjuda en ytlig översikt över potentiellt användbara ämnen eller presentera information på ett oorganiserat sätt och med en osammanhängande skrivstil.
- Tilldela en tredje poäng om utdraget är lämpligt för utbildningsändamål och introducerar nyckelkoncept som är relevanta för skolans eller gymnasieskolans läroplaner. Det är sammanhängande även om det kanske inte är omfattande eller kan inkludera viss överflödig information. Det kan likna en inledande del av en lärobok eller en grundläggande handledning som är lämplig för lärande men har märkbara begränsningar som att behandla komplexa koncept ytligt.
- Ge en fjärde poäng om utdraget är mycket relevant och fördelaktigt för pedagogiska ändamål och uppvisar en klar och konsekvent skrivstil. Det kan vara likt ett kapitel från en lärobok eller en detaljerad handledning som erbjuder omfattande pedagogiskt innehåll, inklusive övningar och lösningar, med minimal irrelevant information. Innehållet är sammanhängande, fokuserat och värdefullt för strukturerat lärande.
- Tilldela en femte poäng om utdraget är enastående i sitt pedagogiska värde, perfekt lämpat för undervisning upp till gymnasieskola. Det följer detaljerad logik, skrivstilen är lätt att följa och erbjuder djupgående och grundliga insikter i ämnet, utan icke-pedagogiskt eller överdrivet komplext innehåll.

Utdraget bör vara på svenska. Om det är till största delen på ett annat språk bör 0 poäng ges.

Utdraget: """,
    """
Efter att ha granskat utdraget:

* Motivera först kortfattat din totala poäng, upp till 100 ord.
* Avsluta ditt svar med poängen i formatet: \"Pedagogiskt värde: <total poäng>\"""",
]

PROMPT_FINEWEB_JSON_NO = [
    """Nedenfor er et utdrag fra en nettside. Vurder om siden har høy pedagogisk verdi og kan være nyttig i en pedagogisk setting for undervisning fra barneskole opp til videregående skole ved hjelp av det additive 5-punkts vurderingssystemet beskrevet nedenfor. Poeng akkumuleres basert på tilfredsstillelse av hvert kriterium:

- Legg til 1 poeng hvis utdraget gir noen grunnleggende informasjon som er relevant for pedagogiske emner, selv om det inkluderer noe irrelevant eller ikke-akademisk innhold som annonser og reklame.
- Legg til et annet poeng hvis utdraget tar for seg visse elementer som er relevante for utdanning, men ikke stemmer overens med utdanningsstandarder. Det kan blande pedagogisk innhold med ikke-pedagogisk materiale, tilby en overfladisk oversikt over potensielt nyttige emner, eller presentere informasjon på en uorganisert måte og med en usammenhengende skrivestil.
- Tildel et tredje poeng hvis utdraget er egnet for pedagogisk bruk og introduserer viktige konsepter som er relevante for skole- eller videregående pensum. Det er sammenhengende, selv om det kanskje ikke er omfattende eller kan inneholde noe overflødig informasjon. Det kan ligne en innledende del av en lærebok eller en grunnleggende veiledning som er egnet for læring, men har merkbare begrensninger som å behandle komplekse konsepter overfladisk.
- Gi et fjerde poeng hvis utdraget er svært relevant og nyttig for pedagogiske formål, og viser en klar og konsekvent skrivestil. Det kan være likt et kapittel fra en lærebok eller en detaljert veiledning som tilbyr betydelig pedagogisk innhold, inkludert øvelser og løsninger, med minimal irrelevant informasjon. Innholdet er sammenhengende, fokusert og verdifullt for strukturert læring.
- Tildel et femte poeng hvis utdraget er enestående i sin pedagogiske verdi, perfekt egnet for undervisning opp til videregående skole. Det følger detaljert resonnement, skrivestilen er lett å følge og tilbyr grundig innsikt i emnet, uten ikke-pedagogisk eller overdrevent komplekst innhold.

Utdraget: """,
    """
Etter å ha undersøkt utdraget:

* Begrunn kort din totale poengsum, opptil 100 ord.
* Tilbakemeldingen skal være i json med feltene \"reason\" and \"educational score\", hvor den første er en tekststreng og den siste en integer.""",
]

PROMPT_FINEWEB_NO = [
    """Nedenfor er et utdrag fra en nettside. Vurder om siden har høy pedagogisk verdi og kan være nyttig i en pedagogisk setting for undervisning fra barneskole opp til videregående skole ved hjelp av det additive 5-punkts vurderingssystemet beskrevet nedenfor. Poeng akkumuleres basert på tilfredsstillelse av hvert kriterium:

- Legg til 1 poeng hvis utdraget gir noen grunnleggende informasjon som er relevant for pedagogiske emner, selv om det inkluderer noe irrelevant eller ikke-akademisk innhold som annonser og reklame.
- Legg til et annet poeng hvis utdraget tar for seg visse elementer som er relevante for utdanning, men ikke stemmer overens med utdanningsstandarder. Det kan blande pedagogisk innhold med ikke-pedagogisk materiale, tilby en overfladisk oversikt over potensielt nyttige emner, eller presentere informasjon på en uorganisert måte og med en usammenhengende skrivestil.
- Tildel et tredje poeng hvis utdraget er egnet for pedagogisk bruk og introduserer viktige konsepter som er relevante for skole- eller videregående pensum. Det er sammenhengende, selv om det kanskje ikke er omfattende eller kan inneholde noe overflødig informasjon. Det kan ligne en innledende del av en lærebok eller en grunnleggende veiledning som er egnet for læring, men har merkbare begrensninger som å behandle komplekse konsepter overfladisk.
- Gi et fjerde poeng hvis utdraget er svært relevant og nyttig for pedagogiske formål, og viser en klar og konsekvent skrivestil. Det kan være likt et kapittel fra en lærebok eller en detaljert veiledning som tilbyr betydelig pedagogisk innhold, inkludert øvelser og løsninger, med minimal irrelevant informasjon. Innholdet er sammenhengende, fokusert og verdifullt for strukturert læring.
- Tildel et femte poeng hvis utdraget er enestående i sin pedagogiske verdi, perfekt egnet for undervisning opp til videregående skole. Det følger detaljert resonnement, skrivestilen er lett å følge og tilbyr grundig innsikt i emnet, uten ikke-pedagogisk eller overdrevent komplekst innhold.

Utdraget: """,
    """
Etter å ha undersøkt utdraget:

* Begrunn kort din totale poengsum, opptil 100 ord.
* Avslutt ditt svar med poeng i formatet: \"Pedagogisk verdi: <total poeng>\"""",
]

PROMPT_FINEWEB_LANG_NO = [
    """Nedenfor er et utdrag fra en nettside. Vurder om siden har høy pedagogisk verdi og kan være nyttig i en pedagogisk setting for undervisning fra barneskole opp til videregående skole ved hjelp av det additive 5-punkts vurderingssystemet beskrevet nedenfor. Poeng akkumuleres basert på tilfredsstillelse av hvert kriterium:

- Legg til 1 poeng hvis utdraget gir noen grunnleggende informasjon som er relevant for pedagogiske emner, selv om det inkluderer noe irrelevant eller ikke-akademisk innhold som annonser og reklame.
- Legg til et annet poeng hvis utdraget tar for seg visse elementer som er relevante for utdanning, men ikke stemmer overens med utdanningsstandarder. Det kan blande pedagogisk innhold med ikke-pedagogisk materiale, tilby en overfladisk oversikt over potensielt nyttige emner, eller presentere informasjon på en uorganisert måte og med en usammenhengende skrivestil.
- Tildel et tredje poeng hvis utdraget er egnet for pedagogisk bruk og introduserer viktige konsepter som er relevante for skole- eller videregående pensum. Det er sammenhengende, selv om det kanskje ikke er omfattende eller kan inneholde noe overflødig informasjon. Det kan ligne en innledende del av en lærebok eller en grunnleggende veiledning som er egnet for læring, men har merkbare begrensninger som å behandle komplekse konsepter overfladisk.
- Gi et fjerde poeng hvis utdraget er svært relevant og nyttig for pedagogiske formål, og viser en klar og konsekvent skrivestil. Det kan være likt et kapittel fra en lærebok eller en detaljert veiledning som tilbyr betydelig pedagogisk innhold, inkludert øvelser og løsninger, med minimal irrelevant informasjon. Innholdet er sammenhengende, fokusert og verdifullt for strukturert læring.
- Tildel et femte poeng hvis utdraget er enestående i sin pedagogiske verdi, perfekt egnet for undervisning opp til videregående skole. Det følger detaljert resonnement, skrivestilen er lett å følge og tilbyr grundig innsikt i emnet, uten ikke-pedagogisk eller overdrevent komplekst innhold.

Utdraget skal være på norsk. Hvis det for det meste er på et annet språk, skal det gis 0 poeng.

Utdraget: """,
    """
Etter å ha undersøkt utdraget:

* Begrunn kort din totale poengsum, opptil 100 ord.
* Avslutt ditt svar med poeng i formatet: \"Pedagogisk verdi: <total poeng>\"""",
]

PROMPT_EN = [
    """Below is an extract from a web page in {language}. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching using the additive 4-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information or world knowledge relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract is suitable for educational purposes and introduces key concepts relevant to school curricula. The text is coherent even if the content is not necessarily comprehensive. It may contain some redundant information.
- Award a third point if the extract is highly relevant for educational purposes and is written in a clear and coherent manner. It may resemble a chapter in a textbook or a tutorial and contains significant educational content, such as exercises and solutions, with minimal irrelevant content.
- Grant a fourth point if the extract is excellent in its educational value and is perfectly suited for teaching. It follows detailed reasoning. The text is easy to follow and offers profound and thoughtful insights into the subject matter, without any non-educational content that uses overly complex terms.

If the text is incoherent, mostly consists of lists without informative content, or only contains pure SEO or marketing material, then give 0 points.

The extract:
""",
    """
After examining the extract:

* First briefly justify your total score, up to 50 words.
* Conclude with the score in the format: \"Educational value: <total points>\"""",
]

PROMPT_FINEWEB = [
    """Below is an extract from a web page in {language}. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:""",
    """
After examining the extract:

* First briefly justify your total score, up to 100 words.
* Conclude with the score using the format: "Educational score: <total points>""",
]


PROMPT_CLEANNESS = {
    "nb": "Nedenfor er et utdrag fra en tekstkilde. Først, vurder 'renheten' til teksten ved å bruke det additive 6-punkts scoringssystemet beskrevet nedenfor, hvor 0 indikerer svært dårlig renhet og 5 indikerer perfekt renhet. Poeng akkumuleres basert på tilfredsstillelsen av hvert kriterium:\n\n1. Legg til 1 poeng hvis teksten er lesbar, men inneholder betydelige feil som skrivefeil, ødelagte setninger eller spredte HTML-tagger.\n2. Legg til ytterligere ett poeng hvis teksten for det meste er lesbar med færre feil, selv om den fortsatt inneholder noen skrivefeil, mindre formateringsproblemer eller sporadiske HTML-tagger.\n3. Tildel et tredje poeng hvis teksten stort sett er ren, med bare mindre feil som sporadiske skrivefeil eller sjeldne formateringsproblemer. Den skal være lett å lese med minimale distraksjoner.\n4. Gi et fjerde poeng hvis teksten er veldig ren, inneholder nesten ingen feil, og eventuelle formateringsproblemer er ubetydelige. Teksten skal flyte jevnt uten noen merkbare avbrudd.\n5. Gi et femte poeng hvis teksten er helt ren, uten feil eller formateringsproblemer i det hele tatt. Den skal være helt fri for skrivefeil, formateringsproblemer eller unødvendige HTML-tagger.\n\nEtter å ha vurdert renheten til det uklippede dokumentet, vurder om trimming av teksten (fjerning av innhold fra begynnelsen, slutten eller begge deler, mens minst 50 % av ordene beholdes) kan forbedre kvaliteten. Vurder deretter renhetsscoren på nytt basert på denne teoretiske trimmingen.\n\nUtdraget:\n\n{content}\n\nEtter å ha undersøkt utdraget:\n\n1. Vurder renheten til det uklippede dokumentet:\n• Begrunn kort din totalscore, opptil 100 ord.\n2. Vurder renhetsscoren på nytt, forutsatt at dokumentet har blitt trimmet (fjerning av opptil 50 % av innholdet fra begynnelsen og/eller slutten):\n• Gi en ny score og begrunnelse hvis aktuelt.\n\nTilbakemelding skal være i JSON med følgende felt:\n\n• reason: En kort begrunnelse for renhetsscoren (string).\n• cleanliness score: Den numeriske scoren som reflekterer renheten til den uklippede teksten.\n• trimmed cleanliness score: Den numeriske scoren som reflekterer renheten til teksten etter teoretisk trimming.\n• trimmed reason: En kort begrunnelse for den nye scoren hvis teksten var trimmet (string).\n\nUansett om teksten er på norsk bokmål eller nynorsk, skal dette ikke påvirke scoren.",
    "en": "Below is an excerpt from a text source. First, assess the 'cleanness' of the text using the additive 6-point scoring system described below, where 0 indicates very poor cleanness and 5 indicates perfect cleanness. Points are accumulated based on the satisfaction of each criterion:\n\n1. Add 1 point if the text is readable but contains significant errors such as typos, broken sentences, or scattered HTML tags.\n2. Add an additional point if the text is mostly readable with fewer errors, though it still contains some typos, minor formatting issues, or occasional HTML tags.\n3. Assign a third point if the text is mostly clean, with only minor errors such as occasional typos or rare formatting issues. It should be easy to read with minimal distractions.\n4. Give a fourth point if the text is very clean, containing almost no errors, and any formatting issues are negligible. The text should flow smoothly without any noticeable interruptions.\n5. Give a fifth point if the text is completely clean, with no errors or formatting issues at all. It should be entirely free of typos, formatting problems, or unnecessary HTML tags.\n\nAfter assessing the cleanness of the untrimmed document, consider whether trimming the text (removing content from the beginning, end, or both, while retaining at least 50% of the words) can improve quality. Then reassess the cleanness score based on this theoretical trimming.\n\nThe excerpt:\n\n{content}\n\nAfter examining the excerpt:\n\n1. Assess the cleanness of the untrimmed document:\n• Briefly justify your total score, up to 100 words.\n2. Reassess the cleanness score assuming the document has been trimmed (removing up to 50% of the content from the beginning and/or end):\n• Provide a new score and justification if applicable.\n\nFeedback should be in JSON with the following fields:\n\n• reason: A brief justification for the cleanness score (string).\n• cleanliness score: The numerical score reflecting the cleanness of the untrimmed text.\n• trimmed cleanliness score: The numerical score reflecting the cleanness of the text after theoretical trimming.\n• trimmed reason: A brief justification for the new score if the text was trimmed (string).",
    "da": "Nedenfor er et uddrag fra en tekstkilde. Først vurder 'renheden' af teksten ved hjælp af det additive 6-punkts scoringssystem beskrevet nedenfor, hvor 0 angiver meget dårlig renhed og 5 angiver perfekt renhed. Point akkumuleres baseret på opfyldelsen af hvert kriterium:\n\n1. Tilføj 1 point, hvis teksten er læsbar, men indeholder betydelige fejl som stavefejl, ødelagte sætninger eller spredte HTML-tags.\n2. Tilføj et yderligere point, hvis teksten for det meste er læsbar med færre fejl, selvom den stadig indeholder nogle stavefejl, mindre formateringsproblemer eller lejlighedsvise HTML-tags.\n3. Tildel et tredje point, hvis teksten hovedsageligt er ren, med kun mindre fejl som lejlighedsvise stavefejl eller sjældne formateringsproblemer. Den skal være let at læse med minimale distraktioner.\n4. Giv et fjerde point, hvis teksten er meget ren, indeholder næsten ingen fejl, og eventuelle formateringsproblemer er ubetydelige. Teksten skal flyde jævnt uden nogen mærkbare afbrydelser.\n5. Giv et femte point, hvis teksten er helt ren, uden fejl eller formateringsproblemer overhovedet. Den skal være helt fri for stavefejl, formateringsproblemer eller unødvendige HTML-tags.\n\nEfter at have vurderet renheden af det uklippede dokument, overvej om trimning af teksten (fjernelse af indhold fra begyndelsen, slutningen eller begge dele, mens mindst 50 % af ordene bevares) kan forbedre kvaliteten. Vurder derefter renhedsscoren igen baseret på denne teoretiske trimning.\n\nUddraget:\n\n{content}\n\nEfter at have undersøgt uddraget:\n\n1. Vurder renheden af det uklippede dokument:\n• Begrund kort din totalscore, op til 100 ord.\n2. Vurder renhedsscoren igen, forudsat at dokumentet er blevet trimmet (fjernelse af op til 50 % af indholdet fra begyndelsen og/eller slutningen):\n• Giv en ny score og begrundelse, hvis det er relevant.\n\nFeedback skal være i JSON med følgende felter:\n\n• reason: En kort begrundelse for renhedsscoren (string).\n• cleanliness score: Den numeriske score, der afspejler renheden af den uklippede tekst.\n• trimmed cleanliness score: Den numeriske score, der afspejler renheden af teksten efter teoretisk trimning.\n• trimmed reason: En kort begrundelse for den nye score, hvis teksten var trimmet (string).",
    "sv": "Nedan följer ett utdrag från en textkälla. Först, bedöm textens 'renhet' med hjälp av det additiva 6-punkts betygssystemet som beskrivs nedan, där 0 indikerar mycket dålig renhet och 5 indikerar perfekt renhet. Poäng ackumuleras baserat på uppfyllelsen av varje kriterium:\n\n1. Lägg till 1 poäng om texten är läsbar men innehåller betydande fel som stavfel, trasiga meningar eller spridda HTML-taggar.\n2. Lägg till ytterligare ett poäng om texten är mestadels läsbar med färre fel, även om den fortfarande innehåller några stavfel, mindre formateringsproblem eller sporadiska HTML-taggar.\n3. Tilldela ett tredje poäng om texten till största delen är ren, med bara mindre fel som sporadiska stavfel eller sällsynta formateringsproblem. Den ska vara lätt att läsa med minimala distraktioner.\n4. Ge ett fjärde poäng om texten är mycket ren, innehåller nästan inga fel, och eventuella formateringsproblem är obetydliga. Texten ska flöda smidigt utan några märkbara avbrott.\n5. Ge ett femte poäng om texten är helt ren, utan fel eller formateringsproblem överhuvudtaget. Den ska vara helt fri från stavfel, formateringsproblem eller onödiga HTML-taggar.\n\nEfter att ha bedömt renheten av det oklippta dokumentet, överväg om trimning av texten (borttagning av innehåll från början, slutet eller båda, medan minst 50% av orden behålls) kan förbättra kvaliteten. Bedöm sedan renhetspoängen på nytt baserat på denna teoretiska trimning.\n\nUtdraget:\n\n{content}\n\nEfter att ha undersökt utdraget:\n\n1. Bedöm renheten av det oklippta dokumentet:\n• Motivera kort din totalsumma, upp till 100 ord.\n2. Bedöm renhetspoängen på nytt, förutsatt att dokumentet har trimmats (borttagning av upp till 50% av innehållet från början och/eller slutet):\n• Ge en ny poäng och motivering om det är tillämpligt.\n\nFeedback ska vara i JSON med följande fält:\n\n• reason: En kort motivering för renhetspoängen (string).\n• cleanliness score: Den numeriska poängen som speglar renheten av den oklippta texten.\n• trimmed cleanliness score: Den numeriska poängen som speglar renheten av texten efter teoretisk trimning.\n• trimmed reason: En kort motivering för den nya poängen om texten trimmades (string).",
}
