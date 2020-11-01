import React, { useState }  from 'react';
import {Button, Col, Form, FormControl, Nav, Navbar, NavDropdown, Row, Table} from "react-bootstrap";
import ResultTable from "../Modules/ResultsTable";
import {useParams, withRouter} from "react-router-dom";
import APIRequest from "../../rest";


let UploadResult = function (props) {
  let { uploadId } = useParams();
  const [data, setData] = useState({})
  let intervalUpdate = function() {
    APIRequest('upload/'+uploadId, {}, '', ).then(
      function (result) {
        setData(result);
      }
    )
  }

  if (data.process_status !== 'failed' && data.process_status !== 'complete') {
    setTimeout(intervalUpdate, 2000);//wait 2 seconds
  }

  if (data.process_status === 'complete') {
    return (
      <div>
        <ResultTable result={data.result}/>
      </div>
    )
  } else if (data.process_status === 'failed') {
    return <h1>Произошла ошибка при обработке данных</h1>
  } else {
    return <h1>Обработка...</h1>
  }

}

export default withRouter(UploadResult);