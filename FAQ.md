# FAQ voor extern:
1. Hoe komen we aan de studentenaantallen zoals we zien bij het prognosetabblad linksonder?
2. Denk na over meer FAQ's.


# Scenario's FAQ voor intern:
1. Inputdata verandert, wat moet er gebeuren?
	Wanneer de inputdata van een voorgaande week verandert (bijvoorbeeld, we zijn nu in week 28 en de data uit week 14 blijkt toch niet te kloppen en is daarom nu aangepast), dan zal alles vanaf die desbetreffende week opnieuw gerund moetten worden. Dit komt omdat het model niet alleen de nieuwste data gebruikt om een voorspelling te doen, maar ook hoe deze nieuwe data zich vergelijkt met de historische data. Dit geldt zowel voor het individuele als het cumulatieve model.
2. De 1-oktobercijfers zijn binnen, wat nu?
	1. Valideren prognose afgelopen jaar\
		Wanneer de 1-oktobercijfers binnen zijn, gebruik het 'update_total_file.py' script om de studentcount bestanden up te daten en het nieuwe predictiejaar toe te voegen.\
	    `uv run update_total_file`\
	2. Hogerejaarsvoorspelling draaien\
		`uv run scripts/standalone/higher_years.py -year NIEUW_COLLEGEJAAR`\
		De output is nu te vinden in je lokale outputfolder ("data/output/lookup_higher_years.xlsx"), verplaats dit naar de inputfolder.
	3. Eerstejaarsvoorspelling (opnieuw) draaien\
		Voorspellingen vanaf week 39 van het nieuwe collegejaar (opnieuw) draaien. Omdat nu de studentenaantallen van het collegejaar ervoor bekend zijn, kunnen deze meegenomen worden in de trainingsdata en zullen de voorspellingen mogelijk veranderen.\
		`uv run main -y NIEUW_COLLEGEJAAR -w 39:HUIDIGE_WEEK`
3. We zijn bij week 39, wat moet er gebeuren?
4. Waarom verbergen we de eerste en laatste paar weken?
5. Denk na over meer scenario's.
