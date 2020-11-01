import React, {useState} from 'react';
import {
  useParams
} from "react-router-dom";
import {Button, Col, Form, Row} from "react-bootstrap";
import APIRequest from "../../rest";
import ClassesSelect from "../Modules/ClassesSelect";

let View = function(props) {
    let { docId } = useParams();
    const [data, setData] = useState({})
    let intervalUpdate = function() {
      APIRequest('view/'+docId, {}, '', ).then(
        function (result) {
          setData(result);
        }
      )
    }

    if (data.process_status !== 'failed' && data.process_status !== 'complete') {
      setTimeout(intervalUpdate, 2000);//wait 2 seconds
    }
    // form
  if (data.process_status === 'complete') {
    return (
      <div>
        <h1 className={'mt-2'}> Просмотр документа {data.doc_src_name} </h1>
        <hr/>
        <Form method={"GET"}>
          <ClassesSelect classes={props.classes} handleChange={null} type={data.doc_cls}/>
          <Form.Group as={Row} controlId="exampleForm.ControlInput1">
            <Form.Label column sm="2">Номер документа</Form.Label>
            <Col sm="10">
              <Form.Control type="text" name={'number'} placeholder="Номер документа" value={data.number} />
            </Col>
          </Form.Group>
          <Form.Group as={Row}  sm="5" controlId="exampleForm.DateFrom">

            <Form.Label column sm="2">Дата подписания</Form.Label>
            <Col sm="10">
              <Form.Control type="date"></Form.Control>
            </Col>
          </Form.Group>
          <Form.Group as={Row} controlId="exampleForm.ControlInput2">
            <Form.Label column sm="2">Выдавший орган</Form.Label>
            <Col sm="10">
              <Form.Control type="text" name={'organization'} placeholder="Выдавший орган" value={data.organization} />
            </Col>
          </Form.Group>

          <Button type={"submit"} variant="outline-success">Сохранить изменения</Button>
        </Form>
        <hr/>
        <iframe className={"col-12"} style={{'height': '800px'}} src={'http://178.154.225.145:5000/api/pdf/'+docId}/>
      </div>
    )
  } else if (data.process_status === 'failed') {
    return <h1>Произошла ошибка при обработке данных</h1>
  } else {
    return <h1>Обработка...</h1>
  }

}

export default View;