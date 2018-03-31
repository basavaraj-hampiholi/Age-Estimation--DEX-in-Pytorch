function [imdbTable, DetectedFaces, SinglesFaces, TwoFaces] = parseIMDB()
    load('imdb.mat');
    imdbS = struct('dob',{},'photo_taken',{}, 'full_path', {}, 'gender', {}, 'name', {}, 'face_location', {}, 'face_score', {}, 'second_face_score', {}, 'age', {});
    for x=1:1:length(imdb.dob)
        imdbS(x).('dob') = imdb.dob(x);
        imdbS(x).('photo_taken') = imdb.photo_taken(x);
        imdbS(x).('full_path') = imdb.full_path(x);
        imdbS(x).('gender') = imdb.gender(x);
        imdbS(x).('name') = imdb.name(x);
        imdbS(x).('face_location') = imdb.face_location(x);
        imdbS(x).('face_score') = imdb.face_score(x);
        imdbS(x).('second_face_score') = imdb.second_face_score(x);
        [age, ~] = datevec(datenum(imdb.photo_taken(x), 7, 1) - imdb.dob(x));
        imdbS(x).('age') = age;

    end
    imdbTable = struct2table(imdbS);
    DetectedFaces = imdbTable(isinf(imdbTable.face_score)==0,:);
    SinglesFaces = DetectedFaces(isnan(DetectedFaces.second_face_score)==1,:);
    TwoFaces = DetectedFaces(isnan(DetectedFaces.second_face_score)==0,:);
end


