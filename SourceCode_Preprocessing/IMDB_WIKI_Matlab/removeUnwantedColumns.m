function result = removeUnwantedColumns(InputTable)
    result = InputTable;
    
    result.dob = [];
    result.photo_taken = [];
    result.name = [];
    result.face_score = [];
    result.second_face_score = [];
end