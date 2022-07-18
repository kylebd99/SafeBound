-- A.1
Select *
from yagofacts as hasGivenName, yagofacts as hasFamilyName, yagofacts as wasBornIn, yagofacts as hasAcademicAdvisor, yagofacts as wasBornIn2, yagofacts as isLocatedIn, yagofacts as isLocatedIn2, yagofacts as type
WHERE hasGivenName.subject=hasFamilyName.subject AND
hasGivenName.subject=wasBornIn.subject AND
wasBornIn.object = isLocatedIn.subject AND
wasBornIn2.object = isLocatedIn2.subject AND
hasAcademicAdvisor.object = wasBornIn2.subject AND
hasAcademicAdvisor.subject= hasGivenName.subject AND
hasGivenName.subject= type.subject AND

type.object = '<wordnet_scientist_110560637>' AND
isLocatedIn2.object='<Germany>' AND
isLocatedIn.object='<Switzerland>' AND

hasGivenName.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND wasBornIn.predicate='<wasBornIn>' AND
hasAcademicAdvisor.predicate='<hasAcademicAdvisor>' AND
wasBornIn2.predicate='<wasBornIn>' AND
isLocatedIn.predicate ='<isLocatedIn>' AND
isLocatedIn2.predicate='<isLocatedIn>' AND
type.predicate='rdf:type';


-- A.2

Select *
from yagofacts as actorType,
yagofacts as movieType,
yagofacts as movieType2,
yagofacts as actedIn,
yagofacts as livesIn,
yagofacts as directed,
yagoFacts as isLocatedIn1,
yagoFacts as isLocatedIn2,
yagoFacts as isLocatedIn3,
yagoFacts as isLocatedIn4
WHERE 
actorType.subject = livesIn.subject AND
livesIn.object = isLocatedIn1.subject AND
isLocatedIn1.object = isLocatedIn2.subject AND
actorType.subject = actedIn.subject AND
actedIn.object = movieType.subject AND
movieType.subject = isLocatedIn3.subject AND
actorType.subject = directed.subject AND
directed.object = movieType2.subject AND
movieType2.subject = isLocatedIn4.subject AND

actorType.object = '<wordnet_actor_109765278>' AND
movieType.object = '<wordnet_movie_106613686>' AND
movieType2.object = '<wordnet_movie_106613686>' AND
isLocatedIn2.object = '<United_States>' AND
isLocatedIn3.object = '<Germany>' AND
isLocatedIn4.object = '<Canada>' AND

actorType.predicate='rdf:type' AND
movieType.predicate='rdf:type' AND
movieType2.predicate='rdf:type' AND
actedIn.predicate='<actedIn>' AND
livesIn.predicate='<livesIn>' AND
directed.predicate='<directed>' AND
isLocatedIn1.predicate = '<isLocatedIn>' AND
isLocatedIn2.predicate = '<isLocatedIn>' AND
isLocatedIn3.predicate = '<isLocatedIn>' AND
isLocatedIn4.predicate = '<isLocatedIn>';

-- A.3
Select *
from yagofacts as actorAthleteType,
yagofacts as politicianType,
yagofacts as wasBornIn,
yagoFacts as isLocatedInState,
yagoFacts as isLocatedInCountry,
yagofacts as hasGivenName,
yagofacts as hasFamilyName
WHERE
politicianType.subject = actorAthleteType.subject AND
politicianType.subject = hasGivenName.subject AND
politicianType.subject = hasFamilyName.subject AND
politicianType.subject = wasBornIn.subject AND
wasBornIn.object = isLocatedInState.subject AND
isLocatedInState.object = isLocatedInCountry.subject AND

actorAthleteType.object IN ('<wordnet_actor_109765278>', '<wordnet_athlete_109820263>') AND
politicianType.object ='<wordnet_politician_110450303>' AND

actorAthleteType.predicate='rdf:type' AND
politicianType.predicate='rdf:type' AND
wasBornIn.predicate='<wasBornIn>' AND
hasGivenName.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND
isLocatedInState.predicate='<isLocatedIn>' AND
isLocatedInCountry.predicate='<isLocatedIn>';

-- B.1
Select *
from 
yagofacts as hasGivenName,
yagofacts as hasGivenName2,
yagofacts as hasFamilyName,
yagofacts as hasFamilyName2,
yagoFacts as isLocatedIn,
yagoFacts as isLocatedIn2,
yagoFacts as livesIn,
yagoFacts as livesIn2,
yagoFacts as actedIn,
yagoFacts as actedIn2
WHERE
hasGivenName.subject = hasFamilyName.subject AND
hasGivenName.subject = livesIn.subject AND
isLocatedIn.subject = livesIn.object AND
hasGivenName.subject = actedIn.subject AND
hasGivenName2.subject = hasFamilyName2.subject AND
hasGivenName2.subject = livesIn2.subject AND
isLocatedIn2.subject = livesIn2.object AND
hasGivenName2.subject = actedIn2.subject AND
actedIn.object = actedIn2.object AND

isLocatedIn.object='<England>' AND
isLocatedIn2.object='<England>' AND

hasGivenName.predicate='<hasGivenName>' AND
hasGivenName2.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND
hasFamilyName2.predicate='<hasFamilyName>' AND
isLocatedIn.predicate='<isLocatedIn>' AND
isLocatedIn2.predicate='<isLocatedIn>' AND
livesIn.predicate='<livesIn>' AND
livesIn2.predicate='<livesIn>' AND
actedIn.predicate='<actedIn>' AND
actedIn2.predicate='<actedIn>';


-- B.2
Select *
from 
yagofacts as hasGivenName,
yagofacts as hasGivenName2,
yagofacts as hasFamilyName,
yagofacts as hasFamilyName2,
yagoFacts as wasBornIn,
yagoFacts as wasBornIn2,
yagoFacts as isMarriedTo
WHERE
hasGivenName.subject = hasFamilyName.subject AND
hasGivenName.subject = wasBornIn.subject AND
hasGivenName.subject = isMarriedTo.subject AND
hasGivenName2.subject = hasFamilyName2.subject AND
hasGivenName2.subject = wasBornIn2.subject AND 
hasGivenName2.subject = isMarriedTo.object AND
wasBornIn.object = wasBornIn2.object AND

hasGivenName.predicate='<hasGivenName>' AND
hasGivenName2.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND
hasFamilyName2.predicate='<hasFamilyName>' AND
wasBornIn.predicate='<wasBornIn>' AND
wasBornIn2.predicate='<wasBornIn>' AND
isMarriedTo.predicate='<isMarriedTo>';

-- B.3 
Select *
from 
yagofacts as hasGivenName,
yagofacts as hasGivenName2,
yagofacts as hasFamilyName,
yagofacts as hasFamilyName2,
yagofacts as type,
yagofacts as type2,
yagofacts as hasWonPrize,
yagofacts as hasWonPrize2,
yagofacts as wasBornIn,
yagofacts as wasBornIn2

WHERE
hasGivenName.subject = hasFamilyName.subject AND
hasGivenName.subject = wasBornIn.subject AND
hasGivenName.subject = hasWonPrize.subject AND
hasGivenName.subject = type.subject AND
hasGivenName2.subject = hasFamilyName2.subject AND
hasGivenName2.subject = wasBornIn2.subject AND
hasGivenName2.subject = hasWonPrize2.subject AND
hasGivenName2.subject = type2.subject AND

hasWonPrize.object = hasWonPrize2.object AND
wasBornIn.object = wasBornIn2.object AND
type.object = '<wordnet_scientist_110560637>' AND
type2.object = '<wordnet_scientist_110560637>' AND

hasGivenName.predicate='<hasGivenName>' AND
hasGivenName2.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND
hasFamilyName2.predicate='<hasFamilyName>' AND
type.predicate = 'rdf:type' AND
type2.predicate = 'rdf:type' AND
hasWonPrize.predicate = '<hasWonPrize>' AND
hasWonPrize2.predicate = '<hasWonPrize>' AND 
wasBornIn.predicate = '<wasBornIn>' AND 
wasBornIn2.predicate = '<wasBornIn>';

-- C.1
Select *
from 
yagofacts as hasGivenName,
yagofacts as hasGivenName2,
yagofacts as hasFamilyName,
yagofacts as hasFamilyName2,
yagofacts as type,
yagofacts as type2,
yagofacts as type3,
yagofacts as type4,
yagofacts as relatedToCity,
yagofacts as relatedToCity2

WHERE
hasGivenName.subject = hasFamilyName.subject AND
hasGivenName.subject = type.subject AND
hasGivenName.subject = relatedToCity.subject AND
relatedToCity.object = type3.subject AND

hasGivenName2.subject = hasFamilyName2.subject AND
hasGivenName2.subject = type2.subject AND
hasGivenName2.subject = relatedToCity2.subject AND
relatedToCity2.object = type4.subject AND

relatedToCity2.object= relatedToCity.object AND


type.object = '<wordnet_scientist_110560637>' AND
type2.object = '<wordnet_scientist_110560637>' AND
type3.object = '<wordnet_city_108524735>' AND
type4.object = '<wordnet_city_108524735>' AND

hasGivenName.predicate='<hasGivenName>' AND
hasGivenName2.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>' AND
hasFamilyName2.predicate='<hasFamilyName>' AND
type.predicate = 'rdf:type' AND
type2.predicate = 'rdf:type' AND
type3.predicate = 'rdf:type' AND
type4.predicate = 'rdf:type';

-- C.2
Select *
from 
yagofacts as hasGivenName,
yagofacts as hasFamilyName,
yagofacts as relatedToParis,
yagofacts as relatedToLondon
WHERE
hasGivenName.subject = hasFamilyName.subject AND
hasGivenName.subject = relatedToParis.subject AND
hasGivenName.subject = relatedToLondon.subject AND

relatedToParis.object = '<Paris>' AND
relatedToLondon.object = '<London>' AND

hasGivenName.predicate='<hasGivenName>' AND
hasFamilyName.predicate='<hasFamilyName>';
