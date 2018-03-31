function [wikiTable, DetectedFaces, SinglesFaces, TwoFaces] = parseWiki()
    load('wiki.mat');
    wikiS = struct('dob',{},'photo_taken',{}, 'full_path', {}, 'gender', {}, 'name', {}, 'face_location', {}, 'face_score', {}, 'second_face_score', {}, 'age', {});
    for x=1:1:length(wiki.dob)
        wikiS(x).('dob') = wiki.dob(x);
        wikiS(x).('photo_taken') = wiki.photo_taken(x);
        wikiS(x).('full_path') = wiki.full_path(x);
        wikiS(x).('gender') = wiki.gender(x);
        wikiS(x).('name') = wiki.name(x);
        wikiS(x).('face_location') = wiki.face_location(x);
        wikiS(x).('face_score') = wiki.face_score(x);
        wikiS(x).('second_face_score') = wiki.second_face_score(x);
        [age, ~] = datevec(datenum(wiki.photo_taken(x), 7, 1) - wiki.dob(x));
        wikiS(x).('age') = age;
    end
        wikiTable = struct2table(wikiS);
        DetectedFaces = wikiTable(isinf(wikiTable.face_score)==0,:);
        SinglesFaces = DetectedFaces(isnan(DetectedFaces.second_face_score)==1,:);
        TwoFaces = DetectedFaces(isnan(DetectedFaces.second_face_score)==0,:);
end
