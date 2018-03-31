% Load the IMDB and WIKI data
[IMDB_All, IMDB_Detected, IMDB_Single, IMDB_Two] = parseIMDB();
[WIKI_All, WIKI_Detected, WIKI_Single, WIKI_Two] = parseWiki();

% Filter the resulting tables  (using Images of single face detection)
filteredTableIMDB = removeUnwantedColumns(IMDB_Single);
filteredTableWIKI = removeUnwantedColumns(WIKI_Single);

% Save filtered tables

writetable(filteredTableIMDB, 'IMDB_SingleFaces.csv')
writetable(filteredTableWIKI, 'WIKI_SingleFaces.csv')